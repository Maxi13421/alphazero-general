import chess
import chess.variant
import chess.engine
import os

def run_fairy_nnue(executable_path, nnue_path):
    # 1. Verify files exist
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return
    if not os.path.exists(nnue_path):
        print(f"Error: NNUE file not found at {nnue_path}")
        return

    # 2. Start the engine
    # Using SimpleEngine for synchronous interaction
    try:
        engine = chess.engine.SimpleEngine.popen_uci(executable_path)
        
        # 3. Configure Engine for NNUE
        # 'Use NNUE' toggles the neural network on
        # 'EvalFile' points to the specific .nnue file
        # 'UCI_Variant' tells Fairy Stockfish the ruleset
        options = {
            "Use NNUE": True,
            "EvalFile": os.path.abspath(nnue_path),
            "Threads": 1,
            "Hash": 512,

        }
        engine.configure(options)
        print(f"Engine configured with NNUE: {nnue_path}")

        # 4. Set up the board (Starting FEN for Crazyhouse)
        board = chess.variant.CrazyhouseBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[PPNBppNB] w KQkq - 0 1")
        print(f"Starting FEN: {board.fen()}")

        # 5. Get Analysis/Move
        # We increase the time/depth slightly to let NNUE process
        result = engine.play(board, chess.engine.Limit(time=2.0))
        
        # Get evaluation info
        info = engine.analyse(board, chess.engine.Limit(depth=15))

        print("--- Result ---")
        print(f"Best Move: {result.move}")
        print(result)
        print(f"Evaluation: {info.get('score').relative}")
        print(info)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Always quit the engine process
        engine.quit()
        print("Engine closed.")

engine_path = "./fairy-stockfish/fairy-stockfish_x86-64-modern"
nnue_path = "./fairy-stockfish/crazyhouse-8ebf84784ad2.nnue"

if __name__ == "__main__":
    
    run_fairy_nnue(engine_path, nnue_path)