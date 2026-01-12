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

cdef int MOVE_PIECE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * BOARD_WIDTH * BOARD_HEIGHT
cdef int PLACE_PAWN_ACTION_SIZE = BOARD_WIDTH * (BOARD_HEIGHT - 2)
cdef int PLACE_OTHER_PIECE_ACTION_SIZE = BOARD_WIDTH * BOARD_HEIGHT * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_CAPTURE_LEFT_ACTION_SIZE = (BOARD_WIDTH - 1) * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_MOVE_FORWARD_ACTION_SIZE = BOARD_WIDTH * (PLACEABLE_PIECE_COUNT - 1)
cdef int PROMOTE_PAWN_CAPTURE_RIGHT_ACTION_SIZE = (BOARD_WIDTH - 1) * (PLACEABLE_PIECE_COUNT - 1)

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
        move[4] = 0
        cdef int quotient
        if(action < MOVE_PIECE_ACTION_SIZE): #normal move
            quotient, move[2] = divmod(action, BOARD_WIDTH)
            quotient, move[3] = divmod(quotient, BOARD_HEIGHT)
            move[1], move[0] = divmod(quotient, BOARD_WIDTH)
            return move
        action -= MOVE_PIECE_ACTION_SIZE
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


    cpdef void move(self, int action):
        self.move_piece(action)

        
        self.last_action = action

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
            


    cpdef int[:] get_valid_moves(self):
        cdef int[:] valid = np.zeros((ACTION_SIZE), dtype=np.intc)

        cdef Py_ssize_t x, y, p
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.pieces[x,y]
                if (piece - EMPTY) >= 0:
                    continue
                if piece == KING_W:
                    self.set_valid_king_moves(valid, x, y)
                elif piece == PAWN_W:
                    self.set_valid_pawn_moves(valid, x, y)
                elif piece == KNIGHT_W:
                    self.set_valid_knight_moves(valid, x, y)
                elif piece == BISHOP_W:
                    self.set_valid_bishop_moves(valid, x, y)
                elif piece == ROOK_W:
                    self.set_valid_rook_moves(valid, x, y)
                elif piece == QUEEN_W:
                    self.set_valid_queen_moves(valid, x, y)


        self.set_valid_pawn_place_moves(valid)
        for p in range(PLACEABLE_PIECE_COUNT - 1):
            self.set_valid_other_piece_place_moves(valid, p)


        return valid

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
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * SQUARE_COUNT
        cdef Py_ssize_t i
        if self.pieces[position_x, position_y + 1] == EMPTY:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_MOVE_FORWARD_OFFSET + position_x * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + (position_y + 1) * BOARD_WIDTH + position_x] = 1
                if position_y == 1 and self.pieces[position_x, 3] == EMPTY:
                    valid[from_offset + 3 * BOARD_WIDTH + position_x] = 1
        if position_x >= 1 and (self.pieces[position_x - 1, position_y + 1] - EMPTY) > 0:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_CAPTURE_LEFT_OFFSET + (position_x - 1) * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + (position_y + 1) * BOARD_WIDTH + position_x - 1] = 1
        if position_x < BOARD_WIDTH - 1 and (self.pieces[position_x + 1, position_y + 1] - EMPTY) > 0:
            if position_y == BOARD_HEIGHT - 2:
                for i in range(PLACEABLE_PIECE_COUNT - 1):
                    valid[PROMOTE_PAWN_CAPTURE_RIGHT_OFFSET + position_x * (PLACEABLE_PIECE_COUNT - 1) + i] = 1
            else:
                valid[from_offset + (position_y + 1) * BOARD_WIDTH + position_x + 1] = 1

        #En passant
        if position_y == BOARD_HEIGHT - 4 and position_x >= 1 and self.pieces[position_x - 1, position_y] == PAWN_B and \
                    self.last_action != -1: 
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x - 1 and BOARD_HEIGHT - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x - 1 and BOARD_HEIGHT - 1 - last_move[3] == position_y:
                valid[from_offset + (position_y + 1) * BOARD_WIDTH + position_x - 1] = 1
        if position_y == BOARD_HEIGHT - 4 and position_x < BOARD_WIDTH - 1 and self.pieces[position_x + 1, position_y] == PAWN_B and \
                    self.last_action != -1:
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x + 1 and BOARD_HEIGHT - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x + 1 and BOARD_HEIGHT - 1 - last_move[3] == position_y:
                valid[from_offset + (position_y + 1) * BOARD_WIDTH + position_x + 1] = 1
        

    cdef void set_valid_king_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * SQUARE_COUNT
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KING_MOVES[move]
            dest_y = position_y + KING_MOVES[move+1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1

        if self.castling_rights[0] and self.pieces[1,0] == EMPTY and self.pieces[2,0] == EMPTY and self.pieces[3,0] == EMPTY \
            and not self.square_is_attacked_by_black(3,0) and not self.square_is_attacked_by_black(4,0):
            valid[from_offset + 2] = 1

        if self.castling_rights[1] and self.pieces[5,0] == EMPTY and self.pieces[6,0] == EMPTY \
            and not self.square_is_attacked_by_black(4,0) and not self.square_is_attacked_by_black(5,0):
            valid[from_offset + 6] = 1



    cdef void set_valid_knight_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * SQUARE_COUNT
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KNIGHT_MOVES[move]
            dest_y = position_y + KNIGHT_MOVES[move+1]
            if dest_x >= BOARD_WIDTH or dest_x < 0 or dest_y >= BOARD_HEIGHT or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1


    cdef void set_valid_queen_moves(self, int[:] valid, int position_x, int position_y):
        self.set_valid_rook_moves(valid, position_x, position_y)
        self.set_valid_bishop_moves(valid, position_x, position_y)


    cdef void set_valid_rook_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * SQUARE_COUNT
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1):
            dest_x -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1):
            dest_x += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_y >= 1):
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break
        
        dest_x = position_x
        dest_y = position_y
        while(dest_y < BOARD_HEIGHT - 1):
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break


    cdef void set_valid_bishop_moves(self, int[:] valid, int position_x, int position_y):
        cdef int from_offset = (BOARD_WIDTH * position_y + position_x) * SQUARE_COUNT
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1 and dest_y >= 1):
            dest_x -= 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y >= 1):
            dest_x += 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x >= 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x -= 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < BOARD_WIDTH - 1 and dest_y < BOARD_HEIGHT - 1):
            dest_x += 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid[from_offset + dest_y * BOARD_WIDTH + dest_x] = 1
                break
            else:
                break


    cpdef int square_is_attacked_by_black(self, int position_x, int position_y):
        return self.square_is_attacked_by_black_pawn(position_x, position_y) or self.square_is_attacked_by_black_rook_or_queen(position_x, position_y) or \
            self.square_is_attacked_by_black_bishop_or_queen(position_x, position_y) or self.square_is_attacked_by_black_king(position_x, position_y)
    
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

        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                value = self.pieces[x,y]
                if value == KING_W:
                    has_white_king = True
                
        if not has_white_king:
            return (True, -player)
        return (False, 0)

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
             + "\n" + str(np.asarray(self.castling_rights)) + "\nLegal moves: " + "".join([str(np.asarray(move)) for move in legal_moves])  +"\n" + str(np.asarray(self.last_action))
