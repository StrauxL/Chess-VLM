import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import json
import chess
import chess.pgn
from dotenv import load_dotenv

# Load environment variables (like WANDB_API_KEY) from .env file if present
load_dotenv()

# =============================================================================
# HARDWARE CONFIGURATION
# User Request: Prioritize NPU / Ryzen iGPU (DirectML). Comment out CUDA.
# =============================================================================

# --- CUDA Config (COMMENTED OUT FOR NOW) ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Running on: {device}")

# --- NPU / DirectML Config ---
device = torch.device('cpu')
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"DirectML is available! Using NPU/iGPU device: {device}")
    else:
        print("WARNING: torch-directml installed but device not found. Defaulting to CPU.")
except ImportError:
    print("WARNING: torch-directml NOT installed. Defaulting to CPU.")
    print("TIP: To use your Ryzen NPU/iGPU, install: pip install torch-directml")
    print("   (Note: CUDA usage is currently commented out in the source code).")


# =============================================================================
# MODEL ARCHITECTURE (Copied from playground.py)
# =============================================================================
config = {
    "n_embd": 384,      
    "n_head": 6,        
    "block_size": 256,  
    "n_layer": 6,       
}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
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
        x_norm = self.ln_1(x)
        q = self.attn_wq(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.attn_wk(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.attn_wv(x_norm).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_wo(attn_out)

        x_norm2 = self.ln_2(x)
        x = x + self.mlp_fc2(F.relu(self.mlp_fc1(x_norm2)))
        return x

def board_to_tensor(board):
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

class VisionTower(nn.Module):
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

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        text_emb = tok_emb + pos_emb

        if vision_boards is not None:
            vis_emb = self.vision_tower(vision_boards)
            x = torch.cat([vis_emb, text_emb], dim=1)
        else:
            x = text_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if vision_boards is not None:
            logits = logits[:, 1:, :]

        return logits, None

# =============================================================================
# WANDB FETCH LOGIC
# =============================================================================
def fetch_model_from_wandb(project_name="playground-gpt", artifact_name="kaggleformer-weights", version="latest"):
    print(f"\n[WandB] Connecting to retrieve {artifact_name}:{version}...")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("[WandB] No WANDB_API_KEY found in environment. Relying on cached login or prompting...")
        wandb.login()

    api = wandb.Api()
    artifact_path = f"{project_name}/{artifact_name}:{version}"

    try:
        artifact = api.artifact(artifact_path)
        print(f"[WandB] Downloading artifact '{artifact_path}'...")
        artifact_dir = artifact.download()

        model_weights_path = os.path.join(artifact_dir, "best_model.pt")
        meta_path = os.path.join(artifact_dir, "meta.json")
        tokenizer_path = os.path.join(artifact_dir, "tokenizer.json")
        if os.path.exists(model_weights_path):
            print(f"[WandB] Successfully downloaded to: {model_weights_path}")
            return model_weights_path, meta_path, tokenizer_path
        else:
            raise FileNotFoundError(f"Expected 'best_model.pt' inside artifact dir: {artifact_dir}")

    except wandb.errors.CommError as e:
        print(f"Failed to communicate with WandB. Ensure project name is correct. Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error fetching from WandB: {e}")
        return None, None, None

# =============================================================================
# INFERENCE LOGIC
# =============================================================================
def main():
    # 1. Fetch
    model_path, meta_path, tokenizer_path = fetch_model_from_wandb()
    if not model_path:
        print("Fallback: Checking if best_model.pt exists locally in the current directory...")
        if os.path.exists("best_model.pt"):
            model_path = "best_model.pt"
            meta_path = "meta.json" if os.path.exists("meta.json") else None
            tokenizer_path = "tokenizer.json" if os.path.exists("tokenizer.json") else None
            print("Using local best_model.pt.")
        else:
            print("Cannot proceed. Model weights not found.")
            return

    # 2. Setup Tokenizer & Model
    print("\n[Init] Setting up BPE tokenizer...")
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        if os.path.exists("tokenizer.json"):
            tokenizer_path = "tokenizer.json"
        elif os.path.exists("d:\\QCSARE\\test_lab\\tokenizer.json"):
            tokenizer_path = "d:\\QCSARE\\test_lab\\tokenizer.json"

    if not meta_path or not os.path.exists(meta_path):
        if os.path.exists("meta.json"):
            meta_path = "meta.json"
        elif os.path.exists("d:\\QCSARE\\test_lab\\meta.json"):
            meta_path = "d:\\QCSARE\\test_lab\\meta.json"

    if meta_path and os.path.exists(meta_path) and tokenizer_path and os.path.exists(tokenizer_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        vocab_size = meta['vocab_size']

        try:
            from tokenizers import Tokenizer
        except ImportError:
            print("Installing tokenizers library...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
            from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded BPE vocabulary ({vocab_size} tokens).")

        def encode(s):
            return tokenizer.encode(s).ids

        def decode(l):
            return tokenizer.decode(l)
    else:
        print("meta.json or tokenizer.json not found. You must train the model first to generate the vocabulary.")
        return

    print(f"[Init] Initializing Original Model (White) & moving to {device}...")
    model_original = GPT(vocab_size, config['n_embd'], config['block_size'], config['n_layer'], config['n_head'])

    # 3. Load Weights
    print(f"[Init] Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model_original.load_state_dict(state_dict)
    model_original.to(device)
    model_original.eval()

    print(f"[Init] Initializing Experimental Model (Black) & moving to {device}...")
    model_experimental = GPT(vocab_size, config['n_embd'], config['block_size'], config['n_layer'], config['n_head'])
    model_experimental.load_state_dict(state_dict)

    # =========================================================================
    # TODO: PRUNING & QUANTIZATION SPACE
    # Apply your pruning, quantization, or LoRA techniques to `model_experimental` 
    # here before moving it to the target device.
    # 
    # Example (Pruning):
    # import torch.nn.utils.prune as prune
    # prune.random_unstructured(model_experimental.blocks[0].attn_wq, name="weight", amount=0.3)
    # =========================================================================

    model_experimental.to(device)
    model_experimental.eval()
    print("[Init] Both models ready for game simulation!\n")

    # 4. Generate Interactive Loop -> Offline Self-Play
    block_size = config['block_size']

    print("\n[Self-Play] Starting offline self-play mode...")

    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    context_str = "1. "
    idx = torch.tensor(encode(context_str), dtype=torch.long, device=device).unsqueeze(0)

    max_moves = 150 # limit game length
    base_temperature = 1.0 # Standard sampling temperature
    max_retries = 50 # Increased retries to give model more chances to find a legal move

    # Top-K sampling helper
    def top_k_logits(logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    print(f"\n[Generation] Playing game locally on {device}... (will save to simulated_game.pgn)")

    start_time = time.time()

    with open("simulated_game.pgn", "w") as pgn_file:
        while not board.is_game_over() and board.fullmove_number <= max_moves:
            # Ensure proper PGN numbering prompt for White
            if board.turn == chess.WHITE and board.fullmove_number > 1:
                prefix = f"{board.fullmove_number}. "
                prefix_idx = torch.tensor(encode(prefix), dtype=torch.long, device=device).unsqueeze(0)
                idx = torch.cat((idx, prefix_idx), dim=1)

            valid_move_found = False

            for attempt in range(max_retries):
                current_context = idx
                generated_chars = []

                # Dynamically generate 8x8x14 spatial board from the environment ONCE per move attempt
                V_cond = board_to_tensor(board).unsqueeze(0).to(device)

                # Dynamic temperature: increase randomness slightly if we keep failing
                # This helps the model escape loops where it deterministically predicts the same illegal move
                current_temperature = base_temperature + (attempt * 0.05) 

                valid_move_str = None
                # Generate up to 12 characters for a move (e.g. "O-O-O# ")
                for _ in range(12): 
                    idx_cond = current_context[:, -block_size:]

                    with torch.no_grad():
                        # Select which model to use based on whose turn it is
                        # White = Original Model, Black = Experimental (Pruned) Model
                        active_model = model_original if board.turn == chess.WHITE else model_experimental
                        logits, _ = active_model(idx_cond, vision_boards=V_cond)

                    logits = logits[:, -1, :] / current_temperature

                    # Apply Top-K sampling to prevent completely garbage characters at high temperatures
                    logits = top_k_logits(logits, k=5) 

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                    next_char = decode([idx_next.item()])

                    # Ignore leading spaces generated by the model if it's the very first token in a new line
                    if not generated_chars and not next_char.strip():
                        continue

                    generated_chars.append(next_char)
                    current_context = torch.cat((current_context, idx_next), dim=1)

                    # BPE pre-tokenizers strip spaces, so we iteratively check if the generated
                    # token sequence thus far constitutes a complete valid chess move.
                    curr_str = "".join(generated_chars).strip()
                    try:
                        temp_move = board.parse_san(curr_str)
                        valid_move_str = curr_str

                        # Handle O-O vs O-O-O ambiguity: don't break early if Queenside castling is still possible
                        if curr_str == "O-O" and any(board.san(m) == "O-O-O" for m in board.legal_moves):
                            continue

                        # If it parses correctly and is unambiguous, safely break early to confirm the move
                        break
                    except ValueError:
                        pass

                move_str = valid_move_str if valid_move_str else "".join(generated_chars).strip()

                if not move_str:
                    continue

                try:
                    # Attempt to parse what the model generated
                    move = board.parse_san(move_str)

                    # Get the strict canonical SAN to keep our context perfect
                    canonical_san = board.san(move)

                    board.push(move)
                    node = node.add_variation(move)
                    valid_move_found = True

                    # Update main context with strictly clean accepted move to prevent hallucination drift
                    clean_suffix = canonical_san + " "
                    suffix_idx = torch.tensor(encode(clean_suffix), dtype=torch.long, device=device).unsqueeze(0)
                    idx = torch.cat((idx, suffix_idx), dim=1)

                    if not board.turn: # Black's turn next, so White just played
                        print(f"{board.fullmove_number}. {canonical_san} ", end="", flush=True)
                    else: # White's turn next, so Black just played
                        print(f"{canonical_san} ", end="", flush=True)

                    # Save PGN progress periodically
                    if board.fullmove_number % 5 == 0:
                        pgn_file.seek(0)
                        pgn_file.write(str(game))
                        pgn_file.truncate()
                        pgn_file.flush()
                    break

                except ValueError:
                    continue

            if not valid_move_found:
                print(f"\n[Terminating] Model failed to find a valid move after {max_retries} attempts. Stopping game.")
                break

        pgn_file.seek(0)
        pgn_file.write(str(game))
        pgn_file.truncate()

    end_time = time.time()
    print(f"\n\nGeneration time: {end_time - start_time:.2f}s")
    print(f"Game finished. Result: {board.result()}. Saved to simulated_game.pgn")

    print("\nUnloading models and clearing memory...")

    # 1. Delete the model and state_dict references
    if 'model_original' in locals():
        del model_original
    if 'model_experimental' in locals():
        del model_experimental
    if 'state_dict' in locals():
        del state_dict

    # 2. Force Python garbage collection to free the references from the iGPU
    import gc
    gc.collect()

    # Note: If you were using CUDA, torch.cuda.empty_cache() would be used here.
    # For DirectML (NPU/iGPU), garbage collection of the tensors is sufficient.

    print("Memory cleared. Battery consumption will return to normal.")

if __name__ == "__main__":
    main()

