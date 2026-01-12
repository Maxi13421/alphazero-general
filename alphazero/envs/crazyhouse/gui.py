import numpy as np
import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI, GameWindow, NUM_BEST_ACTIONS, SideMenuWidget
from alphazero.envs.crazyhouse.crazyhouse import Game
from PySide2.QtCore import Qt

KING_B = 12
EMPTY = 6
PLACABLE_PIECES = 5

class GUI(CustomGUI):

    def __init__(self, *args, **kwargs):
        super().__init__(Game, *args, **kwargs)
        _, self.width, self.height = Game.observation_size()
        self.window = GameWindow(
            self.width,
            self.height + 4,
            cell_size=100,
            title=self.title,
            # image_dir=str(Path(__file__).parent / 'img'),
            evaluator=self.evaluator,
            verbose=True,
            num_best_actions=NUM_BEST_ACTIONS if self.show_hints else 0,
            use_evaluator=(self.evaluator is not None),
            action_to_move=lambda state, action: str(action + 1)
        )
        if self.show_hints:
            self.window.eval_stats_timer.timeout.connect(self._update_draw_actions)
        self.board = self.window.game_board
        self.board.tileClicked.connect(self._tile_click)
        self.board.closing.connect(self.on_window_close)

        self.board.add_circle_pixmap(1, Qt.black)
        self.board.add_circle_pixmap(2, Qt.white)
        
        self.board.add_chesspiece_pixmap(5, "♔", Qt.white)
        self.board.add_chesspiece_pixmap(6, "♙", Qt.white)
        self.board.add_chesspiece_pixmap(7, "♘", Qt.white)
        self.board.add_chesspiece_pixmap(8, "♗", Qt.white)
        self.board.add_chesspiece_pixmap(9, "♖", Qt.white)
        self.board.add_chesspiece_pixmap(10, "♕", Qt.white)
        self.board.add_chesspiece_pixmap(12, "♛", Qt.black)
        self.board.add_chesspiece_pixmap(13, "♜", Qt.black)
        self.board.add_chesspiece_pixmap(14, "♝", Qt.black)
        self.board.add_chesspiece_pixmap(15, "♞", Qt.black)
        self.board.add_chesspiece_pixmap(16, "♟", Qt.black)
        self.board.add_chesspiece_pixmap(17, "♚", Qt.black)

        self.update_state(self._state)

    def _update_draw_actions(self):
        if self.evaluator is None or not self.evaluator.is_running:
            return

        actions = self.evaluator.get_best_actions()
        if not actions:
            return

        self.board.clear_fills()
        self.board.fill_tile(actions[0], 0, Qt.green)
        self.board.fill_tile(actions[-1], 0, Qt.red)

        self.board.update()

    
    def get_board_pos_from_click_pos(self, gui_board_pos):
        if(gui_board_pos[1] == self.height + 2 or gui_board_pos[1] <= 1):
            return None
        if(gui_board_pos[1] == self.height + 3):
            return (gui_board_pos[0], self.height)
        return (gui_board_pos[0], self.height + 1 - gui_board_pos[1])
        
    def get_gui_pos_from_board_coords(self, x, y, is_white = True):
        if(y == self.height):
            return (x, self.height + 3 if is_white else 0)
        return (x, self.height + 1 - y)


    def _tile_click(self, x, y):
        print('[DEBUG] Tile clicked: {} {}'.format(x, y))
        
        if not self.user_input:
            self.board.clear_selection()
            return
        


        board = self._state._board
        player = self._state.player * -2 + 1

        print("Player: {}".format(player))

        def highlight_legals(square):
            if(square == None):
                return False
            
            legals = board.legal_moves(square[0], square[1])
            if not legals:
                return False

            self.board.remove_highlights()
            for end_square in legals:
                sq = self.get_gui_pos_from_board_coords(end_square[0], end_square[1])
                self.board.highlight_tile(sq[0], sq[1])

            self.board.update()
            return True

        def remove_selection():
            print('[DEBUG] Removing selection')
            self.board.clear_selection()
            self.board.remove_highlights()
            self.board.update()


        if self.board.last_selected_tile and self.board.selected_tile and self.get_board_pos_from_click_pos(self.board.last_selected_tile) != None and self.get_board_pos_from_click_pos(self.board.selected_tile) != None:
            from_square = self.get_board_pos_from_click_pos(self.board.last_selected_tile)
            to_square = self.get_board_pos_from_click_pos(self.board.selected_tile)
            move = (from_square, to_square)
            print('[DEBUG] Move: {}'.format(move))
        
            print(np.asarray(board.legal_moves(move[0][0], move[0][1])))
            print([move[1][0], move[1][1], 0])
            if [move[1][0], move[1][1], 0] in board.legal_moves(move[0][0], move[0][1]):
                action = self._state.get_action(move[0][0], move[0][1], move[1][0], move[1][1], 0)
                print('[DEBUG] Move is legal, action: {}'.format(action))
                remove_selection()
                self.on_player_move(action)
            

        elif self.board.selected_tile and highlight_legals(self.get_board_pos_from_click_pos(self.board.selected_tile)):
            print('[DEBUG] Legals highlighted')
            return
        else:
            remove_selection()
            

    
    def show(self):
        self.window.show()

    def close(self):
        self.window.close()
        super().close()

    def undo(self):
        raise NotImplementedError

    def update_state(self, state):
        for x in range(self.width):
            for y in range(self.height):
                pos = self.get_gui_pos_from_board_coords(x,y)
                piece = state._board.pieces[x, y]

                self.board.set_tile(pos[0], pos[1], piece + 5 if piece != EMPTY else None)

        for x in range(PLACABLE_PIECES):
            self.board.set_tile(x, 0, KING_B - x - 1 + 5 if state._board.piece_counts_black[x] > 0 else None)
            
        for x in range(PLACABLE_PIECES):
            self.board.set_tile(x, self.height + 3, x + 1 + 5 if state._board.piece_counts_white[x] > 0 else None)

        

        
        if state.last_action is not None:
            self.board.remove_highlights()
            move = state._board.get_move(state.last_action)
            move_dest = (move[2],move[3] + 2)
            self.board.highlight_tile(move_dest[0], move_dest[1])

        if state.win_state().any():
            self.user_input = False
            self.window.stop_evaluator()
        else:
            self.window.side_menu.update_turn(state.player + 1)
            self.window.run_evaluator(state, block=False)
        
        self.window.update()
        super().update_state(state)


if __name__ == '__main__':
    from PySide2.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    gui = GUI(title='microchess')
    gui.show()
    sys.exit(app.exec_())

