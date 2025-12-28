import pyximport; pyximport.install()

from AlphaZeroGUI.CustomGUI import CustomGUI, GameWindow, NUM_BEST_ACTIONS
from alphazero.envs.microchess.microchess import Game
from PySide2.QtCore import Qt

EMPTY = 5

class GUI(CustomGUI):

    def __init__(self, *args, **kwargs):
        super().__init__(Game, *args, **kwargs)
        _, self.width, self.height = Game.observation_size()
        self.window = GameWindow(
            self.width,
            self.height,
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
        
        self.board.add_chesspiece_pixmap(5, "K", Qt.white)
        self.board.add_chesspiece_pixmap(6, "P", Qt.white)
        self.board.add_chesspiece_pixmap(7, "N", Qt.white)
        self.board.add_chesspiece_pixmap(8, "B", Qt.white)
        self.board.add_chesspiece_pixmap(9, "R", Qt.white)
        self.board.add_chesspiece_pixmap(11, "r", Qt.black)
        self.board.add_chesspiece_pixmap(12, "b", Qt.black)
        self.board.add_chesspiece_pixmap(13, "n", Qt.black)
        self.board.add_chesspiece_pixmap(14, "p", Qt.black)
        self.board.add_chesspiece_pixmap(15, "k", Qt.black)

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

    def get_action(self, move):
        return (self.width * move[0][1] + move[0][0]) * self.width * self.height + (self.width * move[1][1] + move[1][0])
    
    def mirror_corrds_y(self, gui_board_pos):
        return (gui_board_pos[0], self.height - 1 - gui_board_pos[1])


    def _tile_click(self, x, y):
        print('[DEBUG] Tile clicked: {} {}'.format(x, y))
        
        if not self.user_input:
            self.board.clear_selection()
            return
        


        board = self._state._board
        player = self._state.player * -2 + 1

        print("Player: {}".format(player))

        def highlight_legals(square):
            legals = board.legal_moves(square, player)
            if not legals:
                return False

            self.board.remove_highlights()
            for end_square in legals:
                sq = self.mirror_corrds_y(end_square)
                self.board.highlight_tile(sq[0], sq[1])

            self.board.update()
            return True

        def remove_selection():
            print('[DEBUG] Removing selection')
            self.board.clear_selection()
            self.board.remove_highlights()
            self.board.update()


        if self.board.last_selected_tile and self.board.selected_tile:
            from_square = self.mirror_corrds_y(self.board.last_selected_tile)
            to_square = self.mirror_corrds_y(self.board.selected_tile)
            move = (from_square, to_square)
            print('[DEBUG] Move: {}'.format(move))
        
        
            if move[1] in board.legal_moves(from_square, player):
                action = self.get_action(move)
                print('[DEBUG] Move is legal, action: {}'.format(action))
                remove_selection()
                self.on_player_move(action)
            

        elif self.board.selected_tile and highlight_legals(self.mirror_corrds_y(self.board.selected_tile)):
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
                pos = self.mirror_corrds_y((x,y))
                piece = state._board.pieces[pos[0], pos[1]]

                self.board.set_tile(x, y, piece + 5 if piece != EMPTY else None)

        
        if state.last_action is not None:
            # remove previous highlight
            self.board.remove_highlights()
            # highlight the tile where the piece landed
            for y in range(self.height):
                _, move_dest = state.get_move(state.last_action)
                move_dest = self.mirror_corrds_y(move_dest)
                if state._board.pieces[move_dest[0], move_dest[1]] != EMPTY:
                    self.board.highlight_tile(move_dest[0], move_dest[1])
                    break

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
    gui = GUI(title='Connect 4')
    gui.show()
    sys.exit(app.exec_())

