# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True

import numpy as np
#from crazyhouse import BOARD_WIDTH, BOARD_HEIGHT, PLACEABLE_PIECE_COUNT, SQUARE_COUNT, MOVE_PIECE_ACTION_SIZE, PLACE_PAWN_ACTION_SIZE, PLACE_OTHER_PIECE_ACTION_SIZE, PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE, PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE, PROMOTE_PAWN_CAPTURE_RIGHT_ACTION_SIZE, PLACE_PAWN_OFFSET, PLACE_OTHER_PIECE_OFFSET, PROMOTE_PAWN_CAPTURE_LEFT_OFFSET, PROMOTE_PAWN_MOVE_FORWARD_OFFSET, PROMOTE_PAWN_CAPTURE_RIGHT_OFFSET, ACTION_SIZE, KING_W, PAWN_W, KNIGHT_W, BISHOP_W, ROOK_W, QUEEN_W, EMPTY, QUEEN_B, ROOK_B, BISHOP_B, KNIGHT_B, PAWN_B, KING_B, PLAYER_WHITE, PLAYER_BLACK, KNIGHT_MOVES, KING_MOVES

cdef int BOARD_WIDTH = 8
cdef int BOARD_HEIGHT = 8
cdef int PLACEABLE_PIECE_COUNT = 5

cdef int SQUARE_COUNT = BOARD_HEIGHT * BOARD_WIDTH

cdef int MAX_DISTANCE = BOARD_WIDTH - 1 #Assumes squared board
cdef int POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT = 8 * MAX_DISTANCE
cdef int POSSIBLE_KNIGHT_MOVES_FROM_POSITION_COUNT = 8
cdef int POSSIBLE_MOVES_FROM_POSITION_COUNT = POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT + POSSIBLE_KNIGHT_MOVES_FROM_POSITION_COUNT

cdef int MOVE_QUEEN_LIKE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT
cdef int MOVE_KNIGHT_LIKE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * POSSIBLE_KNIGHT_MOVES_FROM_POSITION_COUNT
cdef int MOVE_PIECE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * POSSIBLE_MOVES_FROM_POSITION_COUNT
cdef int PLACE_PAWN_ACTION_SIZE = BOARD_WIDTH * (BOARD_HEIGHT - 2)
cdef int PLACE_OTHER_PIECE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE = (BOARD_WIDTH - 1) * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE = BOARD_WIDTH * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_CAPTURE_RIGHT_ACTION_SIZE = (BOARD_WIDTH - 1) * (PLACEABLE_PIECE_COUNT - 1)

cdef int MOVE_KNIGHT_LIKE_OFFSET = MOVE_QUEEN_LIKE_ACTION_SIZE
cdef int PLACE_PAWN_OFFSET = MOVE_PIECE_ACTION_SIZE
cdef int PLACE_OTHER_PIECE_OFFSET = PLACE_PAWN_OFFSET + PLACE_PAWN_ACTION_SIZE
cdef int PROMOTE_PAWN_CAPTURE_LEFT_OFFSET = PLACE_OTHER_PIECE_OFFSET + PLACE_OTHER_PIECE_ACTION_SIZE
cdef int PROMOTE_PAWN_MOVE_FORWARD_OFFSET = PROMOTE_PAWN_CAPTURE_LEFT_OFFSET + PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE
cdef int PROMOTE_PAWN_CAPTURE_RIGHT_OFFSET = PROMOTE_PAWN_MOVE_FORWARD_OFFSET + PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE

cdef int ACTION_SIZE = PROMOTE_PAWN_CAPTURE_RIGHT_OFFSET + PROMOTE_PAWN_CAPTURE_RIGHT_ACTION_SIZE


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

cdef int PLAYER_WHITE = 1
cdef int PLAYER_BLACK = -1


cdef int[16]  KNIGHT_MOVES =   [1,2 ,  2,1 ,  -1,2 ,  -2, 1 ,  1,-2 ,  2, -1 ,  -1, -2 ,  -2, -1]  
cdef int[16]  KING_MOVES =   [0,1 , 1,1 , 1,0 , 1,-1 , 0,-1 ,  -1,-1 , -1,0 , -1,1 ]

#max pieces on hand
cdef int MAX_PAWNS = 16
cdef int MAX_KNIGHTS = 4
cdef int MAX_BISHOPS = 4
cdef int MAX_ROOKS = 4
cdef int MAX_QUEENS = 2


cdef char[13] int_to_fen_piece_array = [b'K', b'P', b'N', b'B', b'R', b'Q', b'E', b'q', b'r', b'b', b'n', b'p', b'k']
cdef char[8] int_to_file_array = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']




cdef class Board():
    """
    microchess Board.
    """



    cdef public int[:,:] pieces

    cdef public int[:,:] is_piece_promoted

    cdef public int[:] piece_counts_white
    cdef public int[:] piece_counts_black

    cdef public int[:] castling_rights #white_queen, white_king, black_queen, black_king

    cdef public int last_action

    cdef public int[:] valid_moves
    cdef public int are_valid_moves_valid_flag
    #cdef public str history_string



    def __init__(self):
        """Set up initial board configuration."""

        """
        self.pieces = np.array([
            [ROOK_W, EMPTY, EMPTY, PAWN_B, KING_B],
            [BISHOP_W, EMPTY, EMPTY, EMPTY, KNIGHT_B],
            [KNIGHT_W, EMPTY, EMPTY, EMPTY, BISHOP_B],
            [KING_W, PAWN_W, EMPTY, EMPTY, ROOK_B]
        ], dtype=np.intc)
        """
        self.pieces = np.array([
            [ROOK_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, ROOK_B],
            [KNIGHT_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, KNIGHT_B],
            [BISHOP_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, BISHOP_B],
            [QUEEN_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, QUEEN_B],
            [KING_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, KING_B],
            [BISHOP_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, BISHOP_B],
            [KNIGHT_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, KNIGHT_B],
            [ROOK_W, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_B, ROOK_B]
        ], dtype=np.intc)

        self.is_piece_promoted = np.zeros_like(self.pieces, dtype=np.intc)

        self.piece_counts_white = np.zeros(PLACEABLE_PIECE_COUNT, dtype=np.intc)
        self.piece_counts_black = np.zeros(PLACEABLE_PIECE_COUNT, dtype=np.intc)

        self.castling_rights = np.array([True, True, True, True], dtype=np.intc)

        self.last_action = -1

        self.valid_moves = None
        self.are_valid_moves_valid_flag = False


        #self.history_string = ""

    def __getstate__(self):
        return (np.asarray(self.pieces, dtype=np.intc),
            np.asarray(self.is_piece_promoted, dtype=np.intc),
            np.asarray(self.piece_counts_white, dtype=np.intc),
            np.asarray(self.piece_counts_black, dtype=np.intc),
            int(self.castling_rights[0]) << 3 | int(self.castling_rights[1]) << 2, int(self.castling_rights[2]) << 1, int(self.castling_rights[3]),
            self.last_action,
        )

    def __setstate__(self, state):
        self.pieces = np.asarray(state[0], dtype=np.intc)
        self.is_piece_promoted = np.asarray(state[1], dtype=np.intc)
        self.piece_counts_white = np.asarray(state[2], dtype=np.intc)
        self.piece_counts_black = np.asarray(state[3], dtype=np.intc)
        self.castling_rights = np.array([state[4] >> 3, (state[4] >> 2) & 1, (state[4] >> 1) & 1, state[4] & 1], dtype=np.intc)
        self.last_action = state[5]

    
    #move[4] == 0 if no promotion, else it is the piece to promote into
    #if move[1] == BOARD_HEIGHT, move[0] is the piece that get's placed
    cpdef int[:] get_move(self, action : int):
        cdef int[:] move = np.empty(5, dtype=np.intc)  
        cdef int move_from_position, distance
        move[4] = 0
        cdef int quotient
        if(action < MOVE_QUEEN_LIKE_ACTION_SIZE): #normal move
            quotient, move_from_position = divmod(action, POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT)
            move[1], move[0] = divmod(quotient, BOARD_WIDTH)
            quotient, distance = divmod(move_from_position, MAX_DISTANCE)
            move[2] = move[0] + KING_MOVES[quotient * 2] * (distance + 1)
            move[3] = move[1] + KING_MOVES[quotient * 2 + 1] * (distance + 1)
            return move
        action -= MOVE_QUEEN_LIKE_ACTION_SIZE
        if(action < MOVE_KNIGHT_LIKE_ACTION_SIZE): #normal move
            quotient, move_from_position = divmod(action, POSSIBLE_KNIGHT_MOVES_FROM_POSITION_COUNT)
            move[1], move[0] = divmod(quotient, BOARD_WIDTH)
            move[2] = move[0] + KNIGHT_MOVES[move_from_position * 2]
            move[3] = move[1] + KNIGHT_MOVES[move_from_position * 2 + 1]
            return move
        action -= MOVE_KNIGHT_LIKE_ACTION_SIZE
        if(action < PLACE_PAWN_ACTION_SIZE): #Place pawn
            quotient, move[2] = divmod(action, BOARD_WIDTH)
            quotient, move[3] = divmod(quotient, BOARD_HEIGHT)
            move[3] += 1
            move[0] = 1
            move[1] = BOARD_HEIGHT
            return move
        action -= PLACE_PAWN_ACTION_SIZE
        if(action < PLACE_OTHER_PIECE_ACTION_SIZE): #place other piece
            quotient, move[2] = divmod(action, BOARD_WIDTH)
            quotient, move[3] = divmod(quotient, BOARD_HEIGHT)
            move[0] = quotient + 2
            move[1] = BOARD_HEIGHT
            return move
        action -= PLACE_OTHER_PIECE_ACTION_SIZE
        
        #promote pawn (assumes it can promote to everything besides pawn and king)
        if(action < PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE): #capture left
            quotient, move[4]  = divmod(action, PLACEABLE_PIECE_COUNT - 1)
            move[4] += 2
            move[0] = quotient + 1
            move[1] = BOARD_HEIGHT - 2
            move[2] = quotient
            move[3] = BOARD_HEIGHT - 1
            return move
        action -= PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE
        if(action < PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE): #move
            quotient, move[4]  = divmod(action, PLACEABLE_PIECE_COUNT - 1)
            move[4] += 2
            move[0] = quotient
            move[1] = BOARD_HEIGHT - 2
            move[2] = quotient
            move[3] = BOARD_HEIGHT - 1
            return move
        action -= PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE
        if(action < PROMOTE_PAWN_CAPTURE_RIGHT_ACTION_SIZE): #capture right
            quotient, move[4]  = divmod(action, PLACEABLE_PIECE_COUNT - 1)
            move[4] += 2
            move[0] = quotient
            move[1] = BOARD_HEIGHT - 2
            move[2] = quotient + 1
            move[3] = BOARD_HEIGHT - 1
            return move


    cpdef Board clone_board(self):
        cdef Board board = Board()
        board.pieces = np.copy(np.asarray(self.pieces))
        board.is_piece_promoted = np.copy(np.asarray(self.is_piece_promoted))
        board.piece_counts_white = np.copy(np.asarray(self.piece_counts_white))
        board.piece_counts_black = np.copy(np.asarray(self.piece_counts_black))
        board.castling_rights = np.copy(np.asarray(self.castling_rights))
        board.last_action = self.last_action
        return board


    cpdef void move(self, int action):
        #self.history_string += str(self)
        #move = self.get_move(action)
        #self.history_string += f"\n{action} {np.asarray(move)}\n\n\n\n\n"

        self.move_piece(action)        
        self.last_action = action

        self.flip_board()





    cpdef move_piece(self, int action):
        cdef int[:] move = self.get_move(action)
        #print("{} {} {} {}".format(move[0], move[1], move[2], move[3]))
        cdef int piece
        cdef int piece_on_dest
        if(move[1] < BOARD_HEIGHT):
            piece = self.pieces[move[0], move[1]]
            piece_on_dest = self.pieces[move[2], move[3]]
            if(piece_on_dest != EMPTY and piece_on_dest != KING_B):
                if self.is_piece_promoted[move[2], move[3]]:
                    self.is_piece_promoted[move[2], move[3]] = False
                    self.piece_counts_white[0] += 1
                else:
                    self.piece_counts_white[KING_B - piece_on_dest - 1] += 1

            if piece == KING_W:
                self.castling_rights[0] = False
                self.castling_rights[1] = False
            if move[0] == 0 and move[1] == 0:
                self.castling_rights[0] = False
            if move[0] == BOARD_WIDTH - 1 and move[1] == 0:
                self.castling_rights[1] = False  
            if move[2] == 0 and move[3] == BOARD_HEIGHT - 1:
                self.castling_rights[2] = False
            if move[2] == BOARD_WIDTH - 1 and move[3] == BOARD_HEIGHT - 1:
                self.castling_rights[3] = False  

            if piece == KING_W and move[0] - move[2] == 2:
                self.pieces[0,0] = EMPTY
                self.pieces[3,0] = ROOK_W
            if piece == KING_W and move[2] - move[0] == 2:
                self.pieces[7,0] = EMPTY
                self.pieces[5,0] = ROOK_W

            if piece == PAWN_W and move[1] == BOARD_HEIGHT - 4 and self.pieces[move[2], move[1]] == PAWN_B and \
                    (move[0] == move[2] + 1 or move[0] == move[2] - 1) and self.last_action != -1: #En passant
                last_move = self.get_move(self.last_action)
                if last_move[0] == move[2] and BOARD_HEIGHT - 1 - last_move[1] == move[1] + 2 \
                        and last_move[2] == move[2] and BOARD_HEIGHT - 1 - last_move[3] == move[1]:
                    self.piece_counts_white[0] += 1
                    self.pieces[move[2], move[1]] = EMPTY

            if self.is_piece_promoted[move[0], move[1]]:
                self.is_piece_promoted[move[2], move[3]] = True
                self.is_piece_promoted[move[0], move[1]] = False

            if piece == PAWN_W and move[3] == BOARD_HEIGHT - 1:
                self.pieces[move[2], move[3]] = move[4]
                self.is_piece_promoted[move[2], move[3]] = True
            else:   
                self.pieces[move[2], move[3]] = piece
            self.pieces[move[0], move[1]] = EMPTY
        else:
            self.pieces[move[2], move[3]] = move[0]
            self.piece_counts_white[move[0] - 1] -= 1

        self.are_valid_moves_valid_flag = False


    cdef void flip_board(self):
        cdef Py_ssize_t x, y
        for y in range(BOARD_HEIGHT // 2):
            for x in range(BOARD_WIDTH):
                self.pieces[x, y], self.pieces[x, BOARD_HEIGHT - 1 - y] = KING_B - self.pieces[x, BOARD_HEIGHT - 1 - y], KING_B - self.pieces[x, y]
                self.is_piece_promoted[x, y], self.is_piece_promoted[x, BOARD_HEIGHT - 1 - y] = self.is_piece_promoted[x, BOARD_HEIGHT - 1 - y], self.is_piece_promoted[x, y]
        if (BOARD_HEIGHT & 1) == 1:
            for x in range(BOARD_WIDTH):
                self.pieces[x, BOARD_HEIGHT // 2] = KING_B - self.pieces[x, BOARD_HEIGHT // 2]


        self.piece_counts_white, self.piece_counts_black = self.piece_counts_black, self.piece_counts_white

        self.castling_rights[0], self.castling_rights[2] = self.castling_rights[2], self.castling_rights[0]
        self.castling_rights[1], self.castling_rights[3] = self.castling_rights[3], self.castling_rights[1]

        self.are_valid_moves_valid_flag = False


    #for GUI
    cpdef list[int[3]] legal_moves(self, int start_x, int start_y):
        cdef int[:] valid_moves = self.get_valid_moves()
        cdef list legal_moves = []
        cdef Py_ssize_t action
        cdef int[:] move
        for action in range(ACTION_SIZE):
            if valid_moves[action] == 0:
                continue
            move = self.get_move(action)
            if move[0] == start_x and move[1] == start_y:
                legal_moves.append([move[2], move[3], move[4]])

        return legal_moves
            

    cdef void calculate_valid_moves(self):
        self.valid_moves = np.zeros((ACTION_SIZE), dtype=np.intc)

        cdef Py_ssize_t x, y, p, action
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.pieces[x,y]
                if (piece - EMPTY) >= 0:
                    continue
                if piece == KING_W:
                    self.set_valid_king_moves(self.valid_moves, x, y)
                elif piece == PAWN_W:
                    self.set_valid_pawn_moves(self.valid_moves, x, y)
                elif piece == KNIGHT_W:
                    self.set_valid_knight_moves(self.valid_moves, x, y)
                elif piece == BISHOP_W:
                    self.set_valid_bishop_moves(self.valid_moves, x, y)
                elif piece == ROOK_W:
                    self.set_valid_rook_moves(self.valid_moves, x, y)
                elif piece == QUEEN_W:
                    self.set_valid_queen_moves(self.valid_moves, x, y)


        self.set_valid_pawn_place_moves(self.valid_moves)
        for p in range(PLACEABLE_PIECE_COUNT - 1):
            self.set_valid_other_piece_place_moves(self.valid_moves, p)

        cdef tuple king_positions
        cdef int king_x, king_y
        cdef Board board_after_move
        for action in range(ACTION_SIZE):
            if self.valid_moves[action] == 0:
                continue
            board_after_move = self.clone_board()
            board_after_move.move_piece(action)
            king_positions = np.where(np.asarray(board_after_move.pieces) == KING_W)
            if(king_positions[0].size == 0):
                
                ascii_chars = ['♔', '♙', '♘', '♗', '♖', '♕', '□', '♛', '♜', '♝', '♞', '♟', '♚'] 
                result = ""
                for y in range(BOARD_HEIGHT -1, -1, -1):
                    for x in range(BOARD_WIDTH):
                        result += ascii_chars[self.pieces[x,y]] + " "
                    result += "\n"
                print(result)
                print("Board after move")
                result = ""
                for y in range(BOARD_HEIGHT -1, -1, -1):
                    for x in range(BOARD_WIDTH):
                        result += ascii_chars[board_after_move.pieces[x,y]] + " "
                    result += "\n"
                print(result)
            king_x, king_y = king_positions[0][0], king_positions[1][0]
            if board_after_move.square_is_attacked_by_black(king_x, king_y):
                self.valid_moves[action] = 0


    cpdef int[:] get_valid_moves(self):
        if self.are_valid_moves_valid_flag == False:
            self.calculate_valid_moves()
            self.are_valid_moves_valid_flag = True
        return self.valid_moves

    cdef void set_valid_pawn_place_moves(self, int[:] valid):
        if(self.piece_counts_white[0] == 0): return
        cdef Py_ssize_t x, y
        for y in range(1, BOARD_HEIGHT - 1):
                for x in range(BOARD_WIDTH):
                    if(self.pieces[x,y] == EMPTY):
                        valid[PLACE_PAWN_OFFSET + (y - 1) * BOARD_WIDTH + x] = 1

    cdef void set_valid_other_piece_place_moves(self, int[:] valid, int piece_type_minus_two):
        if(self.piece_counts_white[piece_type_minus_two + 1] == 0): return
        cdef Py_ssize_t x, y
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if(self.pieces[x,y] == EMPTY):
                    valid[PLACE_OTHER_PIECE_OFFSET + piece_type_minus_two * SQUARE_COUNT + y * BOARD_WIDTH + x] = 1
        

    cdef void set_valid_pawn_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT
        cdef Py_ssize_t i
        if self.pieces[position_x, position_y + 1] == EMPTY:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_MOVE_FORWARD_OFFSET + position_x * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + 0 * MAX_DISTANCE + (1 - 1)] = 1
                if position_y == 1 and self.pieces[position_x, 3] == EMPTY:
                    valid[from_offset + 0 * MAX_DISTANCE + (2 - 1)] = 1
        if position_x >= 1 and (self.pieces[position_x - 1, position_y + 1] - EMPTY) > 0:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_CAPTURE_LEFT_OFFSET + (position_x - 1) * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + 7 * MAX_DISTANCE + (1 - 1)] = 1
        if position_x < BOARD_WIDTH - 1 and (self.pieces[position_x + 1, position_y + 1] - EMPTY) > 0:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_CAPTURE_RIGHT_OFFSET + position_x * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + 1 * MAX_DISTANCE + (1 - 1)] = 1

        #En passant
        if position_y == BOARD_HEIGHT - 4 and position_x >= 1 and self.pieces[position_x - 1, position_y] == PAWN_B and \
                    self.last_action != -1: 
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x - 1 and BOARD_HEIGHT - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x - 1 and BOARD_HEIGHT - 1 - last_move[3] == position_y:
                valid[from_offset + 7 * MAX_DISTANCE + (1 - 1)] = 1
        if position_y == BOARD_HEIGHT - 4 and position_x < BOARD_WIDTH - 1 and self.pieces[position_x + 1, position_y] == PAWN_B and \
                    self.last_action != -1:
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x + 1 and BOARD_HEIGHT - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x + 1 and BOARD_HEIGHT - 1 - last_move[3] == position_y:
                valid[from_offset + 1 * MAX_DISTANCE + (1 - 1)] = 1
        

    cdef void set_valid_king_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT
        cdef int dest_x, dest_y
        cdef Py_ssize_t direction
        for direction in range(8):
            dest_x = position_x + KING_MOVES[2 * direction]
            dest_y = position_y + KING_MOVES[2 * direction + 1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + direction * MAX_DISTANCE + (1 - 1)] = 1

        if self.castling_rights[0] and self.pieces[1,0] == EMPTY and self.pieces[2,0] == EMPTY and self.pieces[3,0] == EMPTY \
            and not self.square_is_attacked_by_black(3,0) and not self.square_is_attacked_by_black(4,0):
            valid[from_offset + 6 * MAX_DISTANCE + (2 - 1)] = 1

        if self.castling_rights[1] and self.pieces[5,0] == EMPTY and self.pieces[6,0] == EMPTY \
            and not self.square_is_attacked_by_black(4,0) and not self.square_is_attacked_by_black(5,0):
            valid[from_offset + 2 * MAX_DISTANCE + (2 - 1)] = 1



    cdef void set_valid_knight_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = MOVE_KNIGHT_LIKE_OFFSET + (BOARD_WIDTH * position_y + position_x) * POSSIBLE_KNIGHT_MOVES_FROM_POSITION_COUNT
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(8):
            dest_x = position_x + KNIGHT_MOVES[2 * move]
            dest_y = position_y + KNIGHT_MOVES[2 * move + 1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid[from_offset + move] = 1


    cdef void set_valid_queen_moves(self, int[:] valid, int position_x, int position_y):
        self.set_valid_rook_moves(valid, position_x, position_y)
        self.set_valid_bishop_moves(valid, position_x, position_y)


    cdef void set_valid_rook_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1):
            dest_x -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 6 * MAX_DISTANCE + position_x - dest_x - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 6 * MAX_DISTANCE + position_x - dest_x - 1] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1):
            dest_x += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 2 * MAX_DISTANCE + dest_x - position_x - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 2 * MAX_DISTANCE + dest_x - position_x - 1] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_y >= 1):
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 4 * MAX_DISTANCE + position_y - dest_y - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 4 * MAX_DISTANCE + position_y - dest_y - 1] = 1
                break
            else:
                break
        
        dest_x = position_x
        dest_y = position_y
        while(dest_y < BOARD_HEIGHT - 1):
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 0 * MAX_DISTANCE + dest_y - position_y - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 0 * MAX_DISTANCE + dest_y - position_y - 1] = 1
                break
            else:
                break


    cdef void set_valid_bishop_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * POSSIBLE_QUEEN_MOVES_FROM_POSITION_COUNT
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1 and dest_y >= 1):
            dest_x -= 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 5 * MAX_DISTANCE + position_x - dest_x - 1] = 1
            elif self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid[from_offset + 5 * MAX_DISTANCE + position_x - dest_x - 1] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y >= 1):
            dest_x += 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 3 * MAX_DISTANCE + dest_x - position_x - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 3 * MAX_DISTANCE + dest_x - position_x - 1] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x >= 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x -= 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 7 * MAX_DISTANCE + position_x - dest_x - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 7 * MAX_DISTANCE + position_x - dest_x - 1] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x += 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + 1 * MAX_DISTANCE + dest_x - position_x - 1] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + 1 * MAX_DISTANCE + dest_x - position_x - 1] = 1
                break
            else:
                break

    #Doesn't necessarily work if a pawn gets tested as the function doesn't consider en passant
    cpdef int square_is_attacked_by_black(self, int position_x, int position_y):
        return self.square_is_attacked_by_black_pawn(position_x, position_y) or self.square_is_attacked_by_black_rook_or_queen(position_x, position_y) or \
            self.square_is_attacked_by_black_bishop_or_queen(position_x, position_y) or self.square_is_attacked_by_black_king(position_x, position_y) \
             or self.square_is_attacked_by_black_knight(position_x, position_y)
    
    #Not really necessary
    cdef int square_is_attacked_by_black_king(self, int position_x, int position_y):
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KING_MOVES[move]
            dest_y = position_y + KING_MOVES[move+1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] == KING_B:
                return True
        return False

    cdef int square_is_attacked_by_black_pawn(self, int position_x, int position_y):
        if position_x >= 1 and self.pieces[position_x - 1, position_y + 1] == PAWN_B:
            return True
        if position_x < BOARD_WIDTH - 1 and self.pieces[position_x + 1, position_y + 1] == PAWN_B:
            return True

    cdef int square_is_attacked_by_black_knight(self, int position_x, int position_y):
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KNIGHT_MOVES[move]
            dest_y = position_y + KNIGHT_MOVES[move+1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] == KNIGHT_B:
                return True
        return False

    cdef int square_is_attacked_by_black_rook_or_queen(self, int position_x, int position_y):
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1):
            dest_x -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == ROOK_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1):
            dest_x += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == ROOK_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_y >= 1):
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == ROOK_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break
        
        dest_x = position_x
        dest_y = position_y
        while(dest_y < BOARD_HEIGHT - 1):
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == ROOK_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break
        
        return False

    cdef int square_is_attacked_by_black_bishop_or_queen(self, int position_x, int position_y):
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1 and dest_y >= 1):
            dest_x -= 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == BISHOP_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y >= 1):
            dest_x += 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == BISHOP_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x >= 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x -= 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == BISHOP_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x += 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                continue
            elif self.pieces[dest_x,dest_y] == BISHOP_B or self.pieces[dest_x,dest_y] == QUEEN_B:
                return True
            else:
                break

        return False


            

    cpdef tuple[bool,int] get_win_state(self, player : int):
        cdef int value
        cdef bint has_white_king = False
        cdef Py_ssize_t x, y

        if np.asarray(self.get_valid_moves()).any():
            return (False, 0)

        cdef tuple king_positions
        cdef int king_x, king_y
        king_positions = np.where(np.asarray(self.pieces) == KING_W)
        king_x, king_y = king_positions[0][0], king_positions[1][0]
        if(self.square_is_attacked_by_black(king_x, king_y)):
            return (True, -player)  
        return (True, 0)


    

    cpdef str get_fen(self, player):

        cdef Board board
        if player == PLAYER_WHITE:
            board = self
        else:
            board = self.clone_board()
            board.flip_board()

        cdef str fen_string = ""
        cdef int empty_count = 0
        cdef Py_ssize_t x, y, p, i
        for y in range(BOARD_HEIGHT - 1, -1, -1):
            for x in range(BOARD_WIDTH):
                if board.pieces[x, y] == EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_string += str(empty_count)
                        empty_count = 0
                    fen_string += chr(int_to_fen_piece_array[board.pieces[x, y]])
                    if(board.is_piece_promoted[x, y] == 1):
                        fen_string += '~'
            
            if empty_count > 0:
                        fen_string += str(empty_count)
                        empty_count = 0
            if y != 0:
                fen_string += "/"

        fen_string += "["
        for p in range(PLACEABLE_PIECE_COUNT):
            for i in range(board.piece_counts_white[p]):
                fen_string += chr(int_to_fen_piece_array[p + 1])
            for i in range(board.piece_counts_black[p]):
                fen_string += chr(int_to_fen_piece_array[KING_B - (p + 1)])
        fen_string += "]"

        fen_string += " "
        fen_string += "w" if (player == PLAYER_WHITE) else "b"
        fen_string += " "
        cdef int can_castle = False
        if board.castling_rights[1] == 1:
            fen_string += "K"
            can_castle = True
        if board.castling_rights[0] == 1:
            fen_string += "Q"
            can_castle = True
        if board.castling_rights[3] == 1:
            fen_string += "k"
            can_castle = True
        if board.castling_rights[2] == 1:
            fen_string += "q"
            can_castle = True
        if not can_castle:
            fen_string += "-"
        fen_string += " "

        cdef int[:] last_move
        if board.last_action != -1 and board.last_action < MOVE_PIECE_ACTION_SIZE:
            last_move = self.get_move(board.last_action)

            if player == PLAYER_WHITE:
                if last_move[0] == last_move[2] and last_move[3] - last_move[1] == 2 and board.pieces[last_move[2], BOARD_HEIGHT - 1 - last_move[3]] == PAWN_B:
                    fen_string += chr(int_to_file_array[last_move[2]])
                    fen_string += str(BOARD_HEIGHT - 1 - last_move[3] + 1 + 1)
                else:
                    fen_string += "-"
            else:
                if last_move[0] == last_move[2] and last_move[3] - last_move[1] == 2 and board.pieces[last_move[2], last_move[3]] == PAWN_W:
                    fen_string += chr(int_to_file_array[last_move[2]])
                    fen_string += str(last_move[3] - 1 + 1)
                else:
                    fen_string += "-"
        else:
            fen_string += "-"

        return fen_string


    def __str__(self):
        ascii_chars = ['♔', '♙', '♘', '♗', '♖', '♕', '□', '♛', '♜', '♝', '♞', '♟', '♚'] 
        result = ""
        for y in range(BOARD_HEIGHT -1, -1, -1):
            for x in range(BOARD_WIDTH):
                result += ascii_chars[self.pieces[x,y]] + " "
            result += "\n"
        cdef int[:] valid_moves = self.get_valid_moves()
        cdef list legal_moves = []
        cdef Py_ssize_t action
        cdef int[:] move
        for action in range(ACTION_SIZE):
            if valid_moves[action] == 0:
                continue
            move = self.get_move(action)
            legal_moves.append(move)

        return str(np.asarray(self.piece_counts_black)) + "\n" + result \
             + "\n" + str(np.asarray(self.piece_counts_white)) + "\n" + "\n" + str(np.rot90(np.asarray(self.is_piece_promoted), k=1)) \
             + "\n" + str(np.asarray(self.castling_rights)) + "\nLegal moves: " + "".join([str(np.asarray(move)) for move in legal_moves]) + "\n" + " ".join([str(action) for action, valid in enumerate(np.asarray(valid_moves)) if valid])  +"\n" + str(np.asarray(self.last_action))
