from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from multiprocessing import Pool
from multiprocessing.pool import IMapIterator
import os
import pickle
import re
import sys
import numpy as np
import time
from typing import List, Dict, Tuple, Optional

from torch import Tensor
import torch
from alphazero.envs.crazyhouse.crazyhouse import Game
from alphazero.envs.crazyhouse.crazyhouse import (
    KING_W_PY, PAWN_W_PY, KNIGHT_W_PY, BISHOP_W_PY, 
    ROOK_W_PY, QUEEN_W_PY, EMPTY_PY, BOARD_WIDTH_PY, BOARD_HEIGHT_PY
)


MIN_AVG_RATING = 2200
MIN_TIME_CONTROL = 180

class CrazyhousePGNParser:
    def __init__(self):
        self.piece_map = {
            'K': KING_W_PY,
            'P': PAWN_W_PY,
            'N': KNIGHT_W_PY,
            'B': BISHOP_W_PY,
            'R': ROOK_W_PY,
            'Q': QUEEN_W_PY
        }
        
        self.file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        self.rank_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}

    def result_to_onehot(self, result):
        mapping = {
            "1-0": np.array([1, 0, 0], dtype=np.uint8),
            "0-1": np.array([0, 1, 0], dtype=np.uint8),
            "1/2-1/2": np.array([0, 0, 1], dtype=np.uint8)
        }
        return mapping.get(result, None)
    
    def parse_pgn(self, pgn_text: str) -> Tuple[Dict[str, str], List[str], np.ndarray]:
        """Parse PGN text into headers, moves, and result."""
        lines = pgn_text.strip().split('\n')
        
        # Parse headers
        headers = {}
        move_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('[') and line.endswith(']'):
                match = re.match(r'\[(\w+)\s+"([^"]+)"\]', line)
                if match:
                    headers[match.group(1)] = match.group(2)
            else:
                move_start_idx = i
                break
        
        # Parse moves and result
        move_text = ' '.join(lines[move_start_idx:])
        moves, result = self._parse_move_text(move_text)
        
        return headers, moves, result
    
    def _parse_move_text(self, move_text: str) -> Tuple[List[str], np.ndarray]:
        """Extract moves from move text, removing move numbers, annotations, and result."""
        # Remove annotations in curly braces (e.g., { [%eval 0.19] })
        move_text = re.sub(r'\{[^}]*\}', '', move_text)
        
        # Remove move numbers like "1.", "1...", "2."
        move_text = re.sub(r'\d+\.\.\.', '', move_text)  # Remove black move numbers
        move_text = re.sub(r'\d+\.', '', move_text)      # Remove white move numbers
        
        # Remove move quality markers (!, ?, !?, ?!, !!, ??) and unnecessary symbols
        move_text = re.sub(r'[?!+#x]', '', move_text)
        
        # Extract result (1-0, 0-1, 1/2-1/2, or *)
        result_match = re.search(r'(1-0|0-1|1/2-1/2|\*)$', move_text)
        result = self.result_to_onehot(result_match.group(1) if result_match else '*')
        move_text = re.sub(r'(1-0|0-1|1/2-1/2|\*)$', '', move_text)
        
        # Split into individual moves and filter out empty strings
        moves = [move for move in move_text.strip().split() if move]
        
        return moves, result

    
    def parse_move(self, move_str: str, game: Game):
        """Parse a move string and return the action index."""
        index = 0
        if(move_str[index] == 'O'):
            if(move_str == "O-O"):
                return [4, 0, 6, 0, 0], KING_W_PY
            elif(move_str == "O-O-O"):
                return [4, 0, 2, 0, 0], KING_W_PY
        elif(move_str[index].isupper()):
            piece = self.piece_map[move_str[index]]
            index += 1
        else:
            piece = PAWN_W_PY
        
        if(move_str[index] == '@'):
            index += 1
            to_x = self.file_map[move_str[index]]
            index += 1
            to_y = self.rank_map[move_str[index]]
            index += 1
            if(len(move_str) != index):
                raise ValueError(f"Too many characters in move {move_str}")
            move = [piece, BOARD_HEIGHT_PY, to_x, to_y if game.player == 0 else BOARD_HEIGHT_PY - 1 - to_y, 0]
            return move, piece
        else:
            from_x = -1
            from_y = -1
            to_x = -1
            to_y = -1
            promotion = 0
            while(index < len(move_str) and move_str[index].isalnum()):
                if(move_str[index].isalpha()):
                    #print(move_str[index])
                    x = self.file_map[move_str[index]]
                    if(to_x != -1):
                        from_x = to_x
                    to_x = x
                else:
                    y = self.rank_map[move_str[index]]
                    if game.player == 1:
                        y = BOARD_HEIGHT_PY - 1 - y
                    if(to_y != -1):
                        from_y = to_y
                    to_y = y
                index += 1
            if(index < len(move_str) and move_str[index] == '='):
                index += 1
                promotion = self.piece_map[move_str[index]]
                index += 1
            if(len(move_str) != index):
                raise ValueError(f"Too many characters in move {move_str}")
            move = [from_x, from_y, to_x, to_y, promotion]
            return move, piece


                        
                        
        
    def get_move_matches(self, game: Game, move, piece):
        matching_actions = []
        valid_moves = game.valid_moves()
        for action in range(game.action_size()):
            if(valid_moves[action] == 0):
                continue
            board_move = game._board.get_move(action)
            if (move[1] < BOARD_HEIGHT_PY and board_move[1] < BOARD_HEIGHT_PY and game._board.pieces[board_move[0], board_move[1]] == piece) or (move[1] == BOARD_HEIGHT_PY and board_move[1] == BOARD_HEIGHT_PY and board_move[0] == piece):
                if ((move[0] == -1 and board_move[1] < BOARD_HEIGHT_PY) or move[0] == board_move[0]) and ((move[1] == -1 and board_move[1] < BOARD_HEIGHT_PY) or move[1] == board_move[1]) and (move[2] == board_move[2]) \
                    and (move[3] == board_move[3]) and (move[4] == board_move[4]):
                    matching_actions.append(action)
        return matching_actions
    
    
    
    def replay_game(self, pgn_text: str, minimal_average : float = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Replay a game from PGN and return the final game state."""
        headers, moves, result = self.parse_pgn(pgn_text)
        #TODO Maybe filter termination or time control
        if(result is None or int(headers["WhiteElo"]) + int(headers["BlackElo"]) < minimal_average * 2 or headers["Termination"] == "Time forfeit" or headers["TimeControl"] == "-" or int(headers["TimeControl"].split('+')[0]) < MIN_TIME_CONTROL):
            return None
        #print(moves)

        observation_list = []
        policy_list = []
        game = Game()
        
        for i, move_str in enumerate(moves):
            move, piece = self.parse_move(move_str, game)

            matching_actions = self.get_move_matches(game, move, piece)

            if(len(matching_actions) == 0):
                raise ValueError(f"No legal move found for {move_str}")
            if(len(matching_actions) > 1):
                new_matching_actions = []
                for matching_action in matching_actions:
                    clone = game.clone()
                    clone._board.move_piece(matching_action)
                    king_positions = np.where(np.asarray(clone._board.pieces) == KING_W_PY)
                    king_x, king_y = king_positions[0][0], king_positions[1][0]
                    if(not clone._board.square_is_attacked_by_black(king_x, king_y)):
                        new_matching_actions.append(matching_action)
                if(len(new_matching_actions) == 0):
                    raise ValueError(f"First found multiple moves. After checking for attacks on the king no legal move fremained {move_str}")
                if(len(new_matching_actions) > 1):
                    raise ValueError(f"{move_str}/{move} is ambiguous even after checking for attacks on the king with matching actions: {matching_actions} and moves {np.asarray([game._board.get_move(action) for action in matching_actions])}")
                #print(f"Checking for attacks on the king solved ambiguities for move {move_str}/{move} as from matching actions {matching_actions} and moves {np.asarray([game._board.get_move(action) for action in matching_actions])} only action {new_matching_actions} ond move {np.asarray([game._board.get_move(action) for action in new_matching_actions])} remained")
                matching_actions = new_matching_actions
            
            action = matching_actions[0]

            #print(f"{move_str}: {action}")

            observation_list.append(game.observation())
            policy = np.zeros(game.action_size())
            policy[action] = 1
            policy_list.append(policy)
            
            game.play_action(action)
        
        return [(observation_list[i], policy_list[i], result) for i in range(len(observation_list))]
    
    
    def parse_games_multiprocess(self, file_path: str, pool: Pool, processes) -> "Tuple[IMapIterator[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]], int]":
        """
        Parse games using multiple processes (true parallelism).
        
        Args:
            file_path: Path to PGN file
            processes: Number of processes (default: CPU count)
        """
        # Read and split games (fast, do in main process)

        with open(file_path, 'r') as f:
            content = f.read()
        
        game_strings = re.split(r'(?=\[Event)', content)
        game_strings = [g.strip() for g in game_strings if g.strip()]
        

        start_time = time.time()
        checked = 0
        parsed = 0

        print(f"\nParsing {len(game_strings)} games from {file_path} with {processes} processes...")
        
        # Use imap_unordered to get results as they complete
        # chunksize=1 ensures we get updates frequently
        results_iter = pool.imap_unordered(
            self._parse_game_worker, 
            game_strings,
            chunksize=1000
        )
        
        return results_iter, len(game_strings)

    def _parse_game_worker(self, game_str):
        """Worker function for multiprocessing."""
        try:
            return self.replay_game(game_str, MIN_AVG_RATING)
        except Exception as e:
            print(f"Game failed: {e.with_traceback()}")
            return None
    
    def parse_multiple_games_generator(self, file_path: str):
        """Generator version to parse multiple games without loading all into memory."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        game_strings = re.split(r'(?=\[Event)', content)
        game_strings = [g.strip() for g in game_strings if g.strip()]
        
        start_time = time.time()
        for i, game_str in enumerate(game_strings):
            try:
                #print(game_str)
                game, headers, result = self.replay_game(game_str)
                print(f"\r{i}\tAvg: {i/(time.time()-start_time)}", end='')
                yield (game, headers, result)
            except Exception as e:
                print(f"Error parsing game {i+1}: {e}")
                continue

    def get_iter_file(self, iteration: int):
        return f'iteration-{iteration:04d}.pkl'


    def save_game_data(self, game : Game, run_name, directory, processes = 10, batch_size = 100000):
        with Pool(processes=processes) as pool:
            if processes is None:
                processes = os.cpu_count()
            
            

            data_tensor : Tensor = torch.zeros([batch_size, *game.observation_size()])
            policy_tensor : Tensor = torch.zeros([batch_size, game.action_size()])
            value_tensor : Tensor = torch.zeros([batch_size, 3])
            sample_count = 0
            sample_in_current_batch_count = 0
            batch_index = 1
            files = os.listdir(directory)
            for data, policy, value in chain.from_iterable((chain.from_iterable(GameFilterIterator(*self.parse_games_multiprocess(os.path.join(directory,file), pool, processes))) for file in files)): 
                if sample_in_current_batch_count == batch_size:
                    folder = os.path.join("data", run_name)
                    filename = os.path.join(folder, self.get_iter_file(batch_index).replace('.pkl', ''))
                    if not os.path.exists(folder): os.makedirs(folder)

                    torch.save(data_tensor, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    torch.save(policy_tensor, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    torch.save(value_tensor, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                    
                    del data_tensor, policy_tensor, value_tensor
                    
                    data_tensor = torch.zeros([batch_size, *game.observation_size()])
                    policy_tensor = torch.zeros([batch_size, game.action_size()])
                    value_tensor = torch.zeros([batch_size, 3])

                    batch_index += 1
                    sample_in_current_batch_count = 0

                data_tensor[sample_in_current_batch_count] = torch.from_numpy(data)
                policy_tensor[sample_in_current_batch_count] = torch.from_numpy(policy)
                value_tensor[sample_in_current_batch_count] = torch.from_numpy(value)
                sample_in_current_batch_count += 1
                sample_count += 1           
            if sample_in_current_batch_count > 0:
                data_tensor_cut = data_tensor[:sample_in_current_batch_count, :].clone()
                policy_tensor_cut = policy_tensor[:sample_in_current_batch_count, :].clone()
                value_tensor_cut = value_tensor[:sample_in_current_batch_count, :].clone()
                folder = os.path.join("data", run_name)
                filename = os.path.join(folder, self.get_iter_file(batch_index).replace('.pkl', ''))
                if not os.path.exists(folder): os.makedirs(folder)

                torch.save(data_tensor_cut, filename + '-data.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(policy_tensor_cut, filename + '-policy.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(value_tensor_cut, filename + '-value.pkl', pickle_protocol=pickle.HIGHEST_PROTOCOL)
                del data_tensor_cut, policy_tensor_cut, value_tensor_cut

            
            print(sample_count)

            del data_tensor
            del policy_tensor
            del value_tensor

            
        

class GameFilterIterator:
    def __init__(self, iterator, total_games):
        self.iterator = iterator
        self.start_time = time.time()
        self.checked = 0
        self.parsed = 0
        self.positions_parsed = 0
        self.total_games = total_games
    
    def __iter__(self):
        return self
    
    def __next__(self):

        while True:

            
            result = next(self.iterator)

            self.checked += 1
            if result is not None:
                self.parsed += 1
                self.positions_parsed += len(result)
            
            # Calculate average games per second
            elapsed = time.time() - self.start_time
            avg_checked_rate = self.checked / elapsed if elapsed > 0 else 0
            avg_parsed_rate = self.parsed / elapsed if elapsed > 0 else 0
            avg_positions_parsed_rate = self.positions_parsed / elapsed if elapsed > 0 else 0
            games_per_hit = self.checked / self.parsed if self.parsed > 0 else 0
            eta = (self.total_games - self.checked) / avg_checked_rate if avg_checked_rate > 0 else 0
            # Update progress on same line
            print(f"\rChecked: {self.checked}/{self.total_games}\tAvg: {avg_checked_rate:.1f} games/sec, parsed: {self.parsed} Avg {avg_parsed_rate:.1f} games/sec, Avg {games_per_hit:.1f} games per hit, positions parsed: {self.positions_parsed} Avg {avg_positions_parsed_rate:.1f} positions/sec, time: {elapsed:.1f} est. ETA: {eta:.1f}                   ", end='')
            sys.stdout.flush()

            if result is not None:
                return result


# Example usage:
if __name__ == "__main__":
    pgn_example = """[Event "Rated Crazyhouse game"]
[Site "https://lichess.org/SCwcRxrx"]
[White "Seb32"]
[Black "jposthuma"]
[Result "0-1"]
[UTCDate "2016.01.20"]
[UTCTime "03:39:08"]
[WhiteElo "1500"]
[BlackElo "1500"]
[WhiteRatingDiff "-157"]
[BlackRatingDiff "+167"]
[BlackTitle "NM"]
[TimeControl "60+0"]
[Termination "Time forfeit"]
[Variant "Crazyhouse"]

1. e4 d5 2. Nf3 dxe4 3. Ng5 Nf6 4. f3 exf3 5. Nxf3 Bg4 6. Be2 Nc6 7. h3 Bxf3 8. Bxf3 Qd6 9. B@b5 Qg3+ 10. Kf1 P@d7 11. Nc3 N@h2+ 12. Kg1 Nxf3+ 13. Qxf3 B@b6+ 14. P@e3 Qxf3 15. gxf3 Q@g3+ 16. Q@g2 P@f2+ 17. Kf1 Nh5 18. Qxf2 Qxf2+ 19. Kxf2 O-O-O 20. Q@a8+ Nb8 0-1"""
    
    parser = CrazyhousePGNParser()
    #parser.replay_game(pgn_example)

    parser.save_game_data(Game(), "crazyhouse_lichess", "human_games/crazyhouse_lichess", 10, 100000)

    

