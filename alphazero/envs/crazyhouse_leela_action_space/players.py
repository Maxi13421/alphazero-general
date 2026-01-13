import os
import subprocess
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer

import numpy as np

import chess.variant
import chess.engine

import atexit

from alphazero.envs.crazyhouse_leela_action_space.crazyhouse_leela_action_space import (
    KING_W_PY, PAWN_W_PY, KNIGHT_W_PY, BISHOP_W_PY, 
    ROOK_W_PY, QUEEN_W_PY, EMPTY_PY, BOARD_WIDTH_PY, BOARD_HEIGHT_PY
)


class HumanCrazyhousePlayer(BasePlayer):
    @staticmethod
    def is_human() -> bool:
        return True

    def play(self, state: GameState) -> int:
        """
        valid_moves = state.valid_moves()
        print('\nMoves:', [i for (i, valid)
                           in enumerate(valid_moves) if valid])
        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                print('Invalid move')
        """
        input_move = [int(num) for num in input().split(" ")]
        print(input_move)
        move = state.get_action(*input_move)
        print(move)
        print(np.asarray(state._board.get_move(move)))
        return move
    
engine_path = "./fairy-stockfish/fairy-stockfish_x86-64-modern"
nnue_path = "./fairy-stockfish/crazyhouse-8ebf84784ad2.nnue"

file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
rank_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
piece_map = {
            'K': KING_W_PY,
            'P': PAWN_W_PY,
            'N': KNIGHT_W_PY,
            'B': BISHOP_W_PY,
            'R': ROOK_W_PY,
            'Q': QUEEN_W_PY
        }

class FairyStockfishCrazyhousePlayer(BasePlayer):

    def __init__(self, game_cls = None, args = None, verbose = False, elo = 1500):
        super().__init__(game_cls, args, verbose)
        self.board = chess.variant.CrazyhouseBoard()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        atexit.register(self.engine.quit)
        options = {"Use NNUE": True,
            "EvalFile": os.path.abspath(nnue_path),
            "Threads": 1,
            "Hash": 512,
            "UCI_LimitStrength": True,
            "UCI_Elo": elo}
        self.engine.configure(options)


    @staticmethod
    def is_human() -> bool:
        return False

    def play(self, state: GameState) -> int:
        self.board.set_fen(state.get_fen())
        result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
        player = state.player
        print(result.move)
        return self.parse_engine_move(str(result.move), state, player)
    
    
    
    def parse_engine_move(self, move_string : str, state : GameState, player : int) -> int:
        if move_string[1] == '@':
            from_x = piece_map[move_string[0]]
            from_y = BOARD_HEIGHT_PY
        else:
            from_x = file_map[move_string[0]]
            from_y = BOARD_HEIGHT_PY - 1 - rank_map[move_string[1]] if player == 1 else rank_map[move_string[1]]
        to_x = file_map[move_string[2]]
        to_y = BOARD_HEIGHT_PY - 1 - rank_map[move_string[3]] if player == 1 else rank_map[move_string[3]]
        promotion = piece_map[move_string[4].upper()] if len(move_string) > 4 else 0
        print(f"Engine move {from_x} {from_y} {to_x} {to_y} {promotion}")
        action = state.get_action(from_x, from_y, to_x, to_y, promotion)
        print(action)
        return action