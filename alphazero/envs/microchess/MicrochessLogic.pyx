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

PLAYER_WHITE = 1
PLAYER_BLACK = -1


cdef int[16]  KNIGHT_MOVES =   [1,2 ,  2,1 ,  -1,2 ,  -2, 1 ,  1,-2 ,  2, -1 ,  -1, -2 ,  -2, -1]  
cdef int[16]  KING_MOVES =   [0,1 , 1,1 , 1,0 , 1,-1 , 0,-1 ,  -1,-1 , -1,0 , -1,1 ]


cdef class Board():
    """
    microchess Board.
    """



    cdef public int[:,:] pieces

    cdef int width
    cdef int height

    def __init__(self, width, height):
        """Set up initial board configuration."""

        self.width = width
        self.height = height


        self.pieces = np.array([
            [ROOK_W, EMPTY, EMPTY, PAWN_B, KING_B],
            [BISHOP_W, EMPTY, EMPTY, EMPTY, KNIGHT_B],
            [KNIGHT_W, EMPTY, EMPTY, EMPTY, BISHOP_B],
            [KING_W, PAWN_W, EMPTY, EMPTY, ROOK_B]
        ], dtype=np.intc)

    def __getstate__(self):
        return np.asarray(self.pieces)

    def __setstate__(self, state):
        pieces = state
        self.pieces = np.asarray(pieces)

    cpdef int[:] get_move(self, action : int):
        cdef int[4] move
        cdef int remainder
        remainder, move[2] = divmod(action, self.width)
        remainder, move[3] = divmod(remainder, self.height)
        remainder, move[0] = divmod(remainder, self.width)
        remainder, move[1] = divmod(remainder, self.height)
        return move

    cpdef void move(self, int action):
        cdef int[:] move = self.get_move(action)
        cdef int piece = self.pieces[move[0], move[1]]
        if piece == PAWN_W and move[3] == self.height - 1:
            self.pieces[move[2], move[3]] = ROOK_W
        elif piece == PAWN_B and move[3] == 0:
            self.pieces[move[2], move[3]] = ROOK_B
        else:   
            self.pieces[move[2], move[3]] = piece
            
        self.pieces[move[0], move[1]] = EMPTY
        cdef Py_ssize_t x, y
        for y in range(self.height // 2):
            for x in range(self.width):
                self.pieces[x, y], self.pieces[x, self.height - 1 - y] = KING_B - self.pieces[x, self.height - 1 - y], KING_B - self.pieces[x, y]
        if (self.height & 1) == 1:
            for x in range(self.width):
                self.pieces[x, self.height // 2] = KING_B - self.pieces[x, self.height // 2]



    cpdef list[int[2]] legal_moves(self, int start_x, int start_y):
        piece = self.pieces[start_x,start_y]
        if piece - EMPTY >= 0:
            return list() 
        cdef list[int[2]] valid_moves
        if piece > EMPTY:
            piece = KING_B - piece
        if piece == KING_W:
            valid_moves = self.get_valid_king_moves(start_x, start_y)
        elif piece == PAWN_W:
            valid_moves = self.get_valid_pawn_moves(start_x, start_y)
        elif piece == KNIGHT_W:
            valid_moves = self.get_valid_knight_moves(start_x, start_y)
        elif piece == BISHOP_W:
            valid_moves = self.get_valid_bishop_moves(start_x, start_y)
        elif piece == ROOK_W:
            valid_moves = self.get_valid_rook_moves(start_x, start_y)
        return valid_moves


    cpdef int[:] get_valid_moves(self):
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros((self.width*self.width*self.height*self.height), dtype=np.intc)
        cdef list[int[2]] valid_moves
        cdef int from_encoding, index

        cdef Py_ssize_t x, y
        for y in range(self.height):
            for x in range(self.width):
                piece = self.pieces[x,y]
                if (piece - EMPTY) >= 0:
                    continue
                if piece > EMPTY:
                    piece = KING_B - piece
                if piece == KING_W:
                    valid_moves = self.get_valid_king_moves(x,y)
                elif piece == PAWN_W:
                    valid_moves = self.get_valid_pawn_moves(x,y)
                elif piece == KNIGHT_W:
                    valid_moves = self.get_valid_knight_moves(x,y)
                elif piece == BISHOP_W:
                    valid_moves = self.get_valid_bishop_moves(x,y)
                elif piece == ROOK_W:
                    valid_moves = self.get_valid_rook_moves(x,y)

                from_encoding = (self.width * y + x) * self.width * self.height
                for move_dest in valid_moves:
                    index = from_encoding + move_dest[1]*self.width+move_dest[0]
                    valid[index] = 1

        return valid
        

    cdef list[int[2]] get_valid_pawn_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        if self.pieces[position_x, position_y + 1] == EMPTY:
            valid_moves.append([position_x, position_y + 1])
        if position_x >= 1 and (self.pieces[position_x - 1, position_y + 1] - EMPTY) > 0:
            valid_moves.append([position_x - 1, position_y + 1])
        if position_x < self.width - 1 and (self.pieces[position_x + 1, position_y + 1] - EMPTY) > 0:
            valid_moves.append([position_x + 1, position_y + 1])

        return valid_moves

    cdef list[int[2]] get_valid_king_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KING_MOVES[move]
            dest_y = position_y + KING_MOVES[move+1]
            if dest_x >= self.width or dest_x < 0 or dest_y >= self.height or dest_y < 0:
                continue
            if (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
        return valid_moves


    cdef list[int[2]] get_valid_knight_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KNIGHT_MOVES[move]
            dest_y = position_y + KNIGHT_MOVES[move+1]
            if dest_x >= self.width or dest_x < 0 or dest_y >= self.height or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid_moves.append([dest_x, dest_y])
        return valid_moves

    cdef list[int[2]] get_valid_rook_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1):
            dest_x -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < self.width - 1):
            dest_x += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_y >= 1):
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break
        
        dest_x = position_x
        dest_y = position_y
        while(dest_y < self.height - 1):
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break
        
        return valid_moves

    cdef list[int[2]] get_valid_bishop_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        cdef int dest_x = position_x
        cdef int dest_y = position_y
        while(dest_x >= 1 and dest_y >= 1):
            dest_x -= 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif self.pieces[dest_x,dest_y] - EMPTY >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < self.width - 1 and dest_y >= 1):
            dest_x += 1
            dest_y -= 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x >= 1 and dest_y < self.height - 1):
            dest_x -= 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        dest_x = position_x
        dest_y = position_y
        while(dest_x < self.width - 1 and dest_y < self.height - 1):
            dest_x += 1
            dest_y += 1
            if self.pieces[dest_x,dest_y] == EMPTY:
                valid_moves.append([dest_x, dest_y])
            elif (self.pieces[dest_x,dest_y] - EMPTY) >= 0:
                valid_moves.append([dest_x, dest_y])
                break
            else:
                break

        return valid_moves


            

    cpdef tuple[bool,int] get_win_state(self, player : int):
        cdef int value
        cdef bint has_white_king = False
        cdef Py_ssize_t x, y

        for x in range(self.width):
            for y in range(self.height):
                value = self.pieces[x,y]
                if value == KING_W:
                    has_white_king = True
                
        if not has_white_king:
            return (True, -player)
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
