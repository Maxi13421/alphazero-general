from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer

import numpy as np


class HumanMicroChessPlayer(BasePlayer):
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
        move = int(input())
        return move

