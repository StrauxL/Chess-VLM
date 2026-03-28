"""
generate_board_parameters.py

This script processes a chess PGN dataset (e.g., lichess.pgn.zst) and extracts 
various positional and AI-centric metrics based on board states. It outputs a CSV 
dataset mapping a board state (FEN) to multiple quantitative features.

Features extracted:
1. FEN: The input board state.
2. Turn: White or Black.
3. Material_Balance: Raw piece count from White's perspective.
4. Deep_Evaluation: True state evaluation using Stockfish (default depth 15).
5. Hidden_Potential: Compensation/horizon capacity (Deep_Evaluation - Material_Balance).
6. Evaluation_Stability: Volatility of the position (score diff between depth 10 and 15).
7. Mobility: Total legal moves for the current player.
8. Tension_Captures: Number of immediate capturing moves possible.
9. Forcing_Moves: Number of checks available to the current player.

Dependencies:
    pip install chess zstandard
    *Requires a local Stockfish executable.
"""

import os
import argparse
import chess
import chess.engine
import chess.pgn
import io
import csv
import zstandard as zstd

def get_material_balance(board: chess.Board) -> int:
    """Calculates material balance from White's perspective."""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
    return white_material - black_material

def get_positional_parameters(board: chess.Board) -> dict:
    """Calculates lightweight performance parameters for the current player."""
    mobility = board.legal_moves.count()
    captures_available = sum(1 for move in board.legal_moves if board.is_capture(move))
    checks_available = sum(1 for move in board.legal_moves if board.gives_check(move))
    
    return {
        "Mobility": mobility,
        "Tension_Captures": captures_available,
        "Forcing_Moves": checks_available
    }

def calculate_engine_metrics(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int = 15, shallow_depth: int = 10) -> dict:
    """Calculates deep evaluation, hidden potential, and evaluation stability."""
    
    # 1. Get Shallow Engine Evaluation
    info_shallow = engine.analyse(board, chess.engine.Limit(depth=shallow_depth))
    shallow_eval_cp_score = info_shallow["score"].white().score(mate_score=10000)
    # Fallback to mate score if somehow score() returns None
    shallow_eval = shallow_eval_cp_score / 100.0

    # 2. Get Deep Engine Evaluation
    info_deep = engine.analyse(board, chess.engine.Limit(depth=depth))
    deep_eval_cp_score = info_deep["score"].white().score(mate_score=10000)
    deep_eval = deep_eval_cp_score / 100.0
        
    # 3. Parameter Math
    material_balance = get_material_balance(board)
    potential = deep_eval - material_balance
    stability = abs(deep_eval - shallow_eval)
    
    return {
        "Material_Balance": material_balance,
        "Deep_Evaluation": round(deep_eval, 4),
        "Hidden_Potential": round(potential, 4),
        "Evaluation_Stability": round(stability, 4)
    }

def process_pgn_to_csv(pgn_path: str, csv_path: str, engine_path: str, max_positions: int = 10000, deep_depth: int = 15, shallow_depth: int = 10):
    """Parses games from zst PGN, evaluates boards, and exports to CSV."""
    print(f"Loading engine from: {engine_path}")
    
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        print(f"Opening compressed PGN dataset: {pgn_path}")
        with open(pgn_path, 'rb') as fh:
            # Lichess PGNs are heavily compressed, a large window size ensures we don't encounter decompression errors
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                print(f"Writing metrics to: {csv_path}")
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'FEN', 'ASCII_Board', 'Turn', 'Material_Balance', 'Deep_Evaluation', 
                        'Hidden_Potential', 'Evaluation_Stability', 
                        'Mobility', 'Tension_Captures', 'Forcing_Moves'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    positions_processed = 0
                    
                    while positions_processed < max_positions:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            print("Reached the end of the PGN dataset.")
                            break
                        
                        board = game.board()
                        
                        # Loop through mainline moves of the game
                        for move in game.mainline_moves():
                            board.push(move)
                            
                            try:
                                engine_metrics = calculate_engine_metrics(board, engine, depth=deep_depth, shallow_depth=shallow_depth)
                                positional_metrics = get_positional_parameters(board)
                                
                                row = {
                                    'FEN': board.fen(),
                                    'ASCII_Board': str(board),
                                    'Turn': 'White' if board.turn == chess.WHITE else 'Black',
                                }
                                row.update(engine_metrics)
                                row.update(positional_metrics)
                                
                                writer.writerow(row)
                                
                                positions_processed += 1
                                
                                # Periodically flush the file output and report progress
                                if positions_processed % 100 == 0:
                                    print(f"Processed {positions_processed}/{max_positions} positions...")
                                    csvfile.flush() 
                                    
                                if positions_processed >= max_positions:
                                    break
                                    
                            except Exception as e:
                                print(f"Warning: Error processing FEN {board.fen()}: {e}")
                                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create chess parameters CSV dataset from PGN.zst")
    parser.add_argument("--pgn", type=str, default=r"sample_data\lichess.pgn.zst", help="Path to lichess.pgn.zst")
    parser.add_argument("--out", type=str, default=r"sample_data\board_parameters_dataset.csv", help="Output CSV path")
    parser.add_argument("--engine", type=str, default=r"stockfish\stockfish\stockfish-windows-x86-64-avx2.exe", help="Path to Stockfish executable (e.g. C:\\Stockfish\\stockfish.exe)")
    parser.add_argument("--max_pos", type=int, default=1000, help="Max positions to evaluate")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish deep search depth")
    parser.add_argument("--shallow_depth", type=int, default=10, help="Shallow depth used for stability calc")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pgn):
        print(f"Fatal: PGN file not found at '{args.pgn}'. Please verify the path.")
    elif not os.path.exists(args.engine):
        print(f"Fatal: Stockfish engine not found at '{args.engine}'. Please provide valid path.")
    else:
        process_pgn_to_csv(
            pgn_path=args.pgn, 
            csv_path=args.out, 
            engine_path=args.engine, 
            max_positions=args.max_pos,
            deep_depth=args.depth,
            shallow_depth=args.shallow_depth
        )
        print(f"\nDataset successfully generated and saved to '{args.out}'!")
