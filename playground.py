import os
import sys
import math
import random
import time
import datetime
import json
import numpy as np

# NEW IMPORTS FOR PYTORCH
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    class UserSecretsClient:
        def get_secret(self, key):
            return ""

# --- Configuration (NanoGPT defaults) ---
config = {
    "n_embd": 384,      
    "n_head": 6,        
    "block_size": 256,  
    "n_layer": 6,       
    "learning_rate": 1e-3, 
    "num_steps": 100_000,      # Karpathy's NanoGPT default
    "batch_size": 32,       # Reduced from 64 to prevent CUDA OOM on Kaggle T4
    "max_runtime_hours": 12,
}

# Ensure we use GPU / NPU if available!
# If your Asus Zephyrus G14 NPU driver maps to DirectML or a custom torch backend, it will be mapped here automatically.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    print(f"\n[Hardware Target] Initializing model on: {device.upper()}\n")

# Initialize WandB
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("WANDB_API_KEY")

run = None
if wandb_api:
    wandb.login(key=wandb_api)
    run = wandb.init(project="playground-gpt", name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)

random.seed(42)
torch.manual_seed(42)

# --- Data Loading (Chess PGN) ---
import glob
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
local_data_file = os.path.join(script_dir, 'sample_data', 'lichess.pgn.zst')
data_file = local_data_file

if not os.path.exists(data_file):
    # Fallbacks for Kaggle or current directory
    kaggle_search = glob.glob('/kaggle/input/**/*.zst', recursive=True)
    local_search = glob.glob('*.zst')

    if kaggle_search:
        data_file = kaggle_search[0]
        print(f"Found Kaggle dataset: {data_file}")
    elif os.path.exists('/kaggle/working/lichess.pgn.zst'):
        data_file = '/kaggle/working/lichess.pgn.zst'
    elif local_search:
        data_file = local_search[0]
        print(f"Found local dataset: {data_file}")
    else:
        print("Dataset not found. Downloading Lichess standard rated games (Jan 2013)...")
        # Ensure we have requests
        try:
            import requests
        except ImportError:
            print("Installing requests library...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
            import requests

        url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"

        # Determine download path based on environment
        if os.path.exists('/kaggle/working/'):
            dl_path = '/kaggle/working/lichess.pgn.zst'
        else:
            dl_path = local_data_file
            os.makedirs(os.path.dirname(dl_path), exist_ok=True)

        try:
            # Stream download to avoid RAM OOM
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dl_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Download complete: {dl_path}")
            data_file = dl_path
        except Exception as e:
            print(f"Failed to download from Lichess: {e}")
            print("Falling back to Hugging Face dataset (adamkarvonen/chess_games)...")
            try:
                import datasets
            except ImportError:
                print("Installing datasets library...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
                import datasets

            ds = datasets.load_dataset("adamkarvonen/chess_games", split="train", streaming=True)
            text_chunks = []
            total_chars = 0
            for item in ds:
                # Use the first value of the row, which contains the PGN text
                game = list(item.values())[0] 
                if game.strip():
                    text_chunks.append(game)
                    total_chars += len(game)
                if total_chars > 50 * 1024 * 1024:
                    break

            text = "\n".join(text_chunks)
            data_file = None # Indicates text is already loaded into memory

if data_file is not None:
    import zstandard as zstd
    import io
    print("Decompressing and cleaning PGN data (first 500MB for Kaggle Limits)...")
    with open(data_file, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        # Read a chunk instead of the whole file to avoid OOM on huge DBs
        with dctx.stream_reader(f) as reader:
            text = reader.read(500 * 1024 * 1024).decode('utf-8', errors='ignore')
else:
    print("PGN data already loaded into memory via Hugging Face fallback.")

import re
import io
try:
    import chess
    import chess.pgn
except ImportError:
    print("Installing python-chess library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess"])
    import chess
    import chess.pgn

print("Parsing games for Multimodal Dataset...")
pgn_io = io.StringIO(text)
games = []
total_chars_in_vocab = ""

max_games = 50000 # Safe limit for Kaggle RAM
while len(games) < max_games:
    try:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        game_str = game.accept(exporter)
        game_str = re.sub(r'\s+', ' ', game_str).strip()

        if len(game_str) > 20: 
            games.append((game, game_str))
            total_chars_in_vocab += game_str + " "
    except Exception as e:
        continue

# Custom BPE tokenizer for Chess PGN
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except ImportError:
    print("Installing tokenizers library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

# Initialize a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer on the games text in memory
print("Training BPE Tokenizer on PGN data...")
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"], vocab_size=5000)

# We can provide an iterator to train from memory, avoiding the need to write to disk
def get_training_corpus():
    for game_obj, game_str in games:
        yield game_str

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
tokenizer.save("tokenizer.json")
print("BPE Tokenizer trained and saved to tokenizer.json")

vocab_size = tokenizer.get_vocab_size()

def encode(s):
    # .ids gives back the list of integers
    return tokenizer.encode(s).ids 

def decode(l):
    return tokenizer.decode(l)

if __name__ == '__main__':
    print(f"Parsed {len(games)} games into memory.")
    print(f"Vocab size: {vocab_size}")

    meta = {'vocab_size': vocab_size}
    with open('meta.json', 'w') as f:
        json.dump(meta, f)

n_games = len(games)
train_games = games[:int(n_games*0.9)]
val_games = games[int(n_games*0.9):]

def board_to_tensor(board):
    """
    =============================================================================
    EDUCATIONAL EXPLANATION: THE VISION TENSOR (8x8x14)
    =============================================================================
    Why not a 2D array (8x8) of numbers from 1 to 6?
    Because neural networks treat larger numbers mathematically. A King (6) would
    generate a much larger gradient signal than a Pawn (1), which is false! 

    Instead, we use 14 "Channels" (Boolean masks) representing spatial geometry:
    - Channels 0-5  : White Pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    - Channels 6-11 : Black Pieces
    - Channel  12   : Turn to move (All 1s for White, All 0s for Black)
    - Channel  13   : Castling rights (4 specific corners act as flags for Q/K side)

    This provides massive *Inductive Bias*. Instead of the Language Model memorizing 
    which squares are adjacent, the math of the tensor visually proves it!
    =============================================================================
    """
    # Returns (14, 8, 8) Boolean layer feature channels for vision
    tensor = torch.zeros(14, 8, 8, dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            channel = (piece.piece_type - 1)
            if not piece.color: channel += 6
            rank, file = chess.square_rank(square), chess.square_file(square)
            tensor[channel, rank, file] = 1.0
    if board.turn == chess.WHITE: tensor[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[13, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13, 7, 0] = 1.0
    return tensor




# =============================================================================
# PYTORCH MODEL IMPLEMENTATION
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # nn.Parameter tells PyTorch to track gradients for these weights!
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # PyTorch calculates the exact same math, but automates the chained derivative!
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.ln_1 = RMSNorm(n_embd)
        self.attn_wq = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wk = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wv = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)

        self.ln_2 = RMSNorm(n_embd)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # 1. Attention
        x_norm = self.ln_1(x)
        q = self.attn_wq(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hd)
        k = self.attn_wk(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.attn_wv(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch has a highly optimized native function for scaled dot product attention!
        # It handles the causal masking automatically, and computes it exponentially faster than standard matrix multiplication.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_wo(attn_out)

        # 2. MLP
        x_norm2 = self.ln_2(x)
        x = x + self.mlp_fc2(F.relu(self.mlp_fc1(x_norm2)))
        return x

class VisionTower(nn.Module):
    """
    =============================================================================
    EDUCATIONAL EXPLANATION: MULTIMODAL FUSION
    =============================================================================
    How does a Sequence Model (NanoGPT) use a Vision Model (CNN)?

    1. The Conv2d blocks scan the 8x8x14 board state, detecting tactical patterns
       like pins, forks, and pawn chains using sliding spatial filters.
    2. We squash all this visual understanding down into a single massive vector
       of size `n_embd` (384) using a Fully Connected (Linear) layer.
    3. In `GPT.forward`, we will place this `vis_emb` token at the very front 
       of our text sequence (like a [CLS] prepended prompt!). 

    The Transformer's Self-Attention heads now dynamically compute the relationship 
    between:
       (A) The absolute spatial truth of the board (Vision)
       (B) The sequential history of the past moves (Language)
    =============================================================================
    """
    def __init__(self, n_embd):
        super().__init__()
        # 8x8x14 Board State Spatial Extractor
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * 8 * 8, n_embd)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(B, -1)
        x = self.fc(x)
        return x.unsqueeze(1) # Acts as a prompt [CLS] prefix token (B, 1, n_embd)

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.vision_tower = VisionTower(n_embd)

        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, vision_boards=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.wte(idx) # (B, T, C)
        pos_emb = self.wpe(pos) # (T, C)
        text_emb = tok_emb + pos_emb

        if vision_boards is not None:
            vis_emb = self.vision_tower(vision_boards) # (B, 1, C)
            # Prepend the vision spatial context map directly to text tokens!
            x = torch.cat([vis_emb, text_emb], dim=1) # (B, 1+T, C)
        else:
            x = text_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Discard the vision token logit (we don't predict a next vision token directly here)
        if vision_boards is not None:
            logits = logits[:, 1:, :] # Returns back to (B, T, Vocab)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

def get_batch(split):
    """
    =============================================================================
    EDUCATIONAL EXPLANATION: RANDOM MOVE VS MOST PROBABLE
    =============================================================================
    Why do we pick a `random move` during training instead of the most probable?

    In Deep Learning (specifically Supervised Fine-Tuning / Autoregressive training), 
    we need to teach the model how to play in ALL situations: openings, mid-games, 
    and end-games. If we always picked the "most probable" or just starting moves, 
    the model would never learn what to do when there are 5 pieces left on the board!

    So, we look at actual Grandmaster games. We randomly pick an exact moment in that 
    game (the `random move`), rebuild the board up to that moment, feed it to the model, 
    and tell it: "Given this board and this history, predict what the Grandmaster 
    actually played next!"

    This calculates the Loss (error). The model learns to make its "most probable" 
    predictions slowly align with the Grandmaster's real moves over time!
    =============================================================================
    """
    data = train_games if split == 'train' else val_games
    batch_x, batch_y, batch_v = [], [], []

    for _ in range(config['batch_size']):
        game_obj, _ = random.choice(data)
        moves_list = list(game_obj.mainline())
        if not moves_list:
            continue

        # We pick a random position in the game to train the model on predicting
        # the EXACT next move that was made in real life!
        move_idx = random.randint(0, len(moves_list) - 1)

        board = chess.Board()
        history_str = ""
        for i in range(move_idx):
            if board.turn == chess.WHITE:
                history_str += f"{board.fullmove_number}. "
            history_str += f"{board.san(moves_list[i].move)} "
            board.push(moves_list[i].move)

        if board.turn == chess.WHITE:
            history_str += f"{board.fullmove_number}. "

        target_str = f"{board.san(moves_list[move_idx].move)} "
        x_str = history_str + target_str

        # 1) Vision Context: The literal 8x8 spatial grid snapshot right before the move
        batch_v.append(board_to_tensor(board))

        # 2) Text Context: The sequence of tokens
        seq = encode(x_str)
        hist_len = len(encode(history_str))

        full_x = seq[:-1]

        # Target labels mapping!
        # -100 is PyTorch's magic number to IGNORE calculating loss on these tokens.
        # We only want to train the model to predict `target_str`, NOT `history_str`,
        # because the vision tensor is a snapshot of the board AFTER the history is over.
        full_labels = [-100] * len(full_x)
        for i in range(hist_len - 1, len(full_labels)):
            full_labels[i] = seq[i + 1]

        # Truncate to block_size if the sequence is too long
        if len(full_x) > config['block_size']:
            x = full_x[-config['block_size']:]
            y = full_labels[-config['block_size']:]
        else:
            # Pad with 0s for input, and -100s for ignored labels if too short
            pad_len = config['block_size'] - len(full_x)
            x = full_x + [0] * pad_len
            y = full_labels + [-100] * pad_len

        batch_x.append(torch.tensor(x, dtype=torch.long))
        batch_y.append(torch.tensor(y, dtype=torch.long))

    X = torch.stack(batch_x).to(device)
    Y = torch.stack(batch_y).to(device)
    V = torch.stack(batch_v).to(device)
    return X, Y, V

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20) # eval 20 batches
        for k in range(20):
            X, Y, V = get_batch(split)
            logits, loss = model(X, Y, vision_boards=V)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size, config['n_embd'], config['block_size'], config['n_layer'], config['n_head'])
model.to(device)
if __name__ == '__main__':
    print(f"num params: {sum(p.numel() for p in model.parameters())}")

if __name__ == '__main__':
    # --- Training Loop --- 
    # PyTorch natively provides standard optimizers (Adam, AdamW, SGD)
    # This directly replaces our manual velocity/momentum (m_hat / v_hat) tracking code!
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.1)

    num_steps = config['num_steps']
    start_time = time.time()
    max_duration = config['max_runtime_hours'] * 3600
    best_loss = float('inf')

    print(f"Starting training loop on {device}...")

    for step in range(num_steps):
        if time.time() - start_time > max_duration:
            print(f"Reached maximum runtime of {config['max_runtime_hours']} hours. Stopping.")
            break

        # Every 100 steps, evaluate the loss on train and val sets
        if step % 100 == 0:
            losses = estimate_loss()
            print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

            if losses['val'] < best_loss:
                best_loss = losses['val']
                print(f"Saving best model (val_loss {best_loss:.4f})...")
                torch.save(model.state_dict(), "best_model.pt")

            if run:
                wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "step": step})

        X, Y, V = get_batch('train')

        # Forward pass
        logits, loss = model(X, Y, vision_boards=V)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()


    print("Training completed.")

    # --- Save Model ---
    if run:
        print("Logging best model weights to wandb...")
        try:
            artifact = wandb.Artifact("kaggleformer-weights", type="model")
            artifact.add_file("best_model.pt")
            if os.path.exists("meta.json"):
                artifact.add_file("meta.json")
            if os.path.exists("tokenizer.json"):
                artifact.add_file("tokenizer.json")
            run.log_artifact(artifact)
            wandb.finish()
        except FileNotFoundError:
            print("No best_model.pt found to log.")


    # Ensure generation uses the best model weights rather than the last step's weights
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", weights_only=True))
        print("Loaded best_model.pt for generation.")

    # --- Generation ---
    print("\n--- Generating Chess Moves ---")
    model.eval() # Set model to inference mode (affects dropout/batchnorm if used)
    context_str = "1. "
    print(context_str, end="", flush=True)

    block_size = config['block_size']
    tokens_to_generate = 500

    board = chess.Board() # The active generator board

    # Pre-encode context
    idx = torch.tensor(encode(context_str), dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    for _ in range(tokens_to_generate):
        # Crop context to strictly within memory limit
        idx_cond = idx[:, -block_size:]

        # Inject the current visual reality of the board
        V_cond = board_to_tensor(board).unsqueeze(0).to(device)

        with torch.no_grad(): # Disable autograd during generation for massive speedup!
            logits, _ = model(idx_cond, vision_boards=V_cond)

        # Get only the last token's distributions
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # Sample using PyTorch's native function
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Decode specifically the single new token and print it
        next_text = decode([idx_next.item()])
        print(next_text, end="", flush=True)

    print("\n")
