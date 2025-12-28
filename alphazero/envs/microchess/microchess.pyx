# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.microchess.MicrochessLogic import Board

import numpy as np

BOARD_WIDTH = 4
BOARD_HEIGHT = 5


NUM_PLAYERS = 2
MAX_TURNS = 60
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 11 if MULTI_PLANE_OBSERVATION else 1

KING_W = 0
PAWN_W = 1
KNIGHT_W = 2
BISHOP_W = 3
ROOK_W = 4
EMPTY = 5
ROOK_B = 6
BISHOP_B = 7
KNIGHT_B = 8
PAWN_B = 9
KING_B = 10


class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board(BOARD_WIDTH, BOARD_HEIGHT)

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns

    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game._turns = self.turns
        game.last_action = self.last_action
        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return True

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return BOARD_HEIGHT * BOARD_HEIGHT * BOARD_WIDTH * BOARD_WIDTH

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, BOARD_WIDTH, BOARD_HEIGHT

    def get_move(self, action : int) -> tuple[tuple[int, int], tuple[int, int]]:
        cdef int new_x, new_y, old_x, old_y, remainder
        remainder, new_x = divmod(action, BOARD_WIDTH)
        remainder, new_y = divmod(remainder, BOARD_HEIGHT)
        remainder, old_x = divmod(remainder, BOARD_WIDTH)
        remainder, old_y = divmod(remainder, BOARD_HEIGHT)
        return ((old_x, old_y),(new_x, new_y))

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves(self.player * -2 + 1))

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.move(self.get_move(action))
        self._update_turn()

    def win_state(self) -> np.ndarray:
        result = [False] * 3
        game_over, player = self._board.get_win_state()

        if self._turns >= MAX_TURNS - 1:
            game_over = True

        if game_over:
            index = -1
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True


        return np.array(result, dtype=np.uint8)

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            king_w = np.where(pieces == KING_W, 1, 0)
            pawn_w = np.where(pieces == PAWN_W, 1, 0)
            knight_w = np.where(pieces == KNIGHT_W, 1, 0)
            bishop_w = np.where(pieces == BISHOP_W, 1, 0)
            rook_w = np.where(pieces == ROOK_W, 1, 0)
            king_b = np.where(pieces == KING_B, 1, 0)
            pawn_b = np.where(pieces == PAWN_B, 1, 0)
            knight_b = np.where(pieces == KNIGHT_B, 1, 0)
            bishop_b = np.where(pieces == BISHOP_B, 1, 0)
            rook_b = np.where(pieces == ROOK_B, 1, 0)
            colour = np.full_like(pieces, self.player)
            return np.array([king_w, pawn_w, knight_w, bishop_w, rook_w, king_b, pawn_b, knight_b, bishop_b, rook_b, colour], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    #Not used
    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[:, ::-1]
        return [(self.clone(), pi), (new_state, pi[::-1])]
    


def display(board, action=None):
    if action:
        print(f'Action: {action}, Move: {action + 1}')
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
