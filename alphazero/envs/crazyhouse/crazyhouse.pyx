# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.crazyhouse.CrazyhouseLogic import Board

import numpy as np

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
PLACEABLE_PIECES = 5


NUM_PLAYERS = 2
MAX_TURNS = 100
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 29 if MULTI_PLANE_OBSERVATION else 1


#max pieces on hand
MAX_PAWNS = 16
MAX_KNIGHTS = 4
MAX_BISHOPS = 4
MAX_ROOKS = 4
MAX_QUEENS = 2

cdef int KING_W = 0
cdef int PAWN_W = 1
cdef int KNIGHT_W = 2
cdef int BISHOP_W = 3
cdef int ROOK_W = 4
cdef int QUEEN_W = 5
cdef int EMPTY = 6
cdef int QUEEN_B = 7
cdef int ROOK_B = 8
cdef int BISHOP_B = 9
cdef int KNIGHT_B = 10
cdef int PAWN_B = 11
cdef int KING_B = 12


class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())

    @staticmethod
    def _get_board():
        return Board(BOARD_WIDTH, BOARD_HEIGHT, PLACEABLE_PIECES)

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + self._board.is_piece_promoted.tobytes() + self._board.piece_counts_white.tobytes() + self._board.piece_counts_black.tobytes() \
         + self._board.castling_rights.tobytes() + self._board.last_action.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns \
             and self._board.is_piece_promoted == other._board.is_piece_promoted \
             and self._board.piece_counts_white == other._board.piece_counts_white and self._board.piece_counts_black == other._board.piece_counts_black \
             and self._board.castling_rights == other._board.castling_rights and self._board.last_action == other._board.last_action

    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._board.is_piece_promoted = np.copy(np.asarray(self._board.is_piece_promoted))
        game._board.piece_counts_white = np.copy(np.asarray(self._board.piece_counts_white))
        game._board.piece_counts_black = np.copy(np.asarray(self._board.piece_counts_black))
        game._board.castling_rights = np.copy(np.asarray(self._board.castling_rights))
        game._board.last_action = self._board.last_action
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
        return (BOARD_HEIGHT * BOARD_WIDTH + PLACEABLE_PIECES) * BOARD_HEIGHT * BOARD_WIDTH

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, BOARD_WIDTH, BOARD_HEIGHT

    

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves())

    def play_action(self, action: int) -> None:
        self._board.move(action)
        super().play_action(action)
        self._update_turn()

    def win_state(self) -> np.ndarray:
        result = [False] * 3
        game_over, player = self._board.get_win_state(-2 * self.player + 1)

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

    #maybe add turns
    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            king_w = np.where(pieces == KING_W, 1, 0)
            pawn_w = np.where(pieces == PAWN_W, 1, 0)
            knight_w = np.where(pieces == KNIGHT_W, 1, 0)
            bishop_w = np.where(pieces == BISHOP_W, 1, 0)
            rook_w = np.where(pieces == ROOK_W, 1, 0)
            queen_w = np.where(pieces == QUEEN_W, 1, 0)
            king_b = np.where(pieces == KING_B, 1, 0)
            pawn_b = np.where(pieces == PAWN_B, 1, 0)
            knight_b = np.where(pieces == KNIGHT_B, 1, 0)
            bishop_b = np.where(pieces == BISHOP_B, 1, 0)
            rook_b = np.where(pieces == ROOK_B, 1, 0)
            queen_b = np.where(pieces == QUEEN_B, 1, 0)
            pawn_count_w = np.full_like(pieces, self._board.piece_counts_white[0] / MAX_PAWNS)
            knight_count_w = np.full_like(pieces, self._board.piece_counts_white[1] / MAX_KNIGHTS)
            bishop_count_w = np.full_like(pieces, self._board.piece_counts_white[2] / MAX_BISHOPS)
            rook_count_w = np.full_like(pieces, self._board.piece_counts_white[3] / MAX_ROOKS)
            queen_count_w = np.full_like(pieces, self._board.piece_counts_white[4] / MAX_QUEENS)
            pawn_count_b = np.full_like(pieces, self._board.piece_counts_black[0] / MAX_PAWNS)
            knight_count_b = np.full_like(pieces, self._board.piece_counts_black[1] / MAX_KNIGHTS)
            bishop_count_b = np.full_like(pieces, self._board.piece_counts_black[2] / MAX_BISHOPS)
            rook_count_b = np.full_like(pieces, self._board.piece_counts_black[3] / MAX_ROOKS)
            queen_count_b = np.full_like(pieces, self._board.piece_counts_black[4] / MAX_QUEENS)
            castling_queen_w = np.full_like(pieces, self._board.castling_rights[0])
            castling_king_w = np.full_like(pieces, self._board.castling_rights[1])
            castling_queen_b = np.full_like(pieces, self._board.castling_rights[2])
            castling_king_b = np.full_like(pieces, self._board.castling_rights[3])

            is_piece_promoted = np.asarray(self._board.is_piece_promoted)

            en_passantable_pawn = np.zeros_like(pieces)
            if(self._board.last_action != -1 and self._board.last_action < BOARD_HEIGHT * BOARD_WIDTH * BOARD_HEIGHT * BOARD_WIDTH):
                last_move = self._board.get_move(self._board.last_action)
                if(pieces[last_move[2], BOARD_HEIGHT - 1 - last_move[3]] == PAWN_B and last_move[3] - last_move[1] == 2):
                    en_passantable_pawn[last_move[2], BOARD_HEIGHT - 1 - last_move[3]] = 1

            colour = np.full_like(pieces, self.player)
            return np.array([king_w, pawn_w, knight_w, bishop_w, rook_w, queen_w, \
                king_b, pawn_b, knight_b, bishop_b, rook_b, queen_b, \
                pawn_count_w, knight_count_w, bishop_count_w, rook_count_w, queen_count_w, \
                pawn_count_b, knight_count_b, bishop_count_b, rook_count_b, queen_count_b, \
                castling_queen_w, castling_king_w, castling_queen_b, castling_king_b, \
                is_piece_promoted, en_passantable_pawn, colour, \
                ], dtype=np.float32)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def get_action(self, old_x, old_y, new_x, new_y):
        return (self.width * old_y + old_x) * self.width * self.height + (self.width * new_y + new_x)

    #Not used, no symmetries when castling is possible or you want to focus on single starting position
    def symmetries(self, pi : np.ndarray) -> List[Tuple['Game', np.ndarray]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[::-1, :]
        cdef int a_l_l, a_l_r, a_r_l, a_r_r
        cdef Py_ssize_t old_x, old_y, new_x, new_y
        new_pi = np.empty_like(pi)
        for old_x in range(BOARD_WIDTH // 2):
            for old_y in range(BOARD_HEIGHT):
                for new_x in range(BOARD_WIDTH // 2): #Assumes width is even
                    for new_y in range(BOARD_HEIGHT):
                        a_l_l = self.get_action(old_x, old_y, new_x, new_y)
                        a_l_r = self.get_action(old_x, old_y, BOARD_WIDTH - 1 - new_x, new_y)
                        a_r_l = self.get_action(BOARD_WIDTH - 1 - new_x, old_y, new_x, new_y)
                        a_r_r = self.get_action(BOARD_WIDTH - 1 - new_x, old_y, BOARD_WIDTH - 1 - new_x, new_y)
                        new_pi[a_l_l] = pi[a_r_r]
                        new_pi[a_r_r] = pi[a_l_l]
                        new_pi[a_l_r] = pi[a_r_l]
                        new_pi[a_r_l] = pi[a_l_r]

                
        return [(self.clone(), pi), (new_state, new_pi)]
    


def display(board, action=None):
    if action:
        print(f'Action: {action}')
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")
