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


KNIGHT_MOVES = [(1,2), (2,1), (-1,2), (-2, 1), (1,-2), (2, -1), (-1, -2), (-2, -1)]
KING_MOVES = [(0,1),(1,1),(1,0),(1,-1),(0,-1), (-1,-1),(-1,0),(-1,1)]


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
        ])

    def __getstate__(self):
        return np.asarray(self.pieces)

    def __setstate__(self, state):
        pieces = state
        self.pieces = np.asarray(pieces)

    cpdef void move(self, move : tuple[tuple[int,int],tuple[int, int]]):
        cdef int piece = self.pieces[move[0][0], move[0][1]]
        if piece == PAWN_W and move[1][1] == self.height - 1:
            self.pieces[move[1][0], move[1][1]] = ROOK_W
        elif piece == PAWN_B and move[1][1] == 0:
            self.pieces[move[1][0], move[1][1]] = ROOK_B
        else:   
            self.pieces[move[1][0], move[1][1]] = piece
            
        self.pieces[move[0][0], move[0][1]] = EMPTY

    cpdef list[tuple[int, int]] legal_moves(self, start : tuple[int, int], player : int):
        x, y = start[0], start[1]
        piece = self.pieces[x,y]
        if (piece - EMPTY) * player >= 0:
            return list() 
        cdef list[tuple[int, int]] valid_moves
        if piece > EMPTY:
            piece = KING_B - piece
        if piece == KING_W:
            valid_moves = self.get_valid_king_moves((x,y), player)
        elif piece == PAWN_W:
            valid_moves = self.get_valid_pawn_moves((x,y), player)
        elif piece == KNIGHT_W:
            valid_moves = self.get_valid_knight_moves((x,y), player)
        elif piece == BISHOP_W:
            valid_moves = self.get_valid_bishop_moves((x,y), player)
        elif piece == ROOK_W:
            valid_moves = self.get_valid_rook_moves((x,y), player)
        return valid_moves


    cpdef int[:] get_valid_moves(self, player : int):
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros((self.width*self.width*self.height*self.height), dtype=np.intc)
        cdef list[tuple[int, int]] valid_moves

        cdef Py_ssize_t x, y
        for y in range(self.height):
            for x in range(self.width):
                piece = self.pieces[x,y]
                if (piece - EMPTY) * player >= 0:
                    continue
                if piece > EMPTY:
                    piece = KING_B - piece
                if piece == KING_W:
                    valid_moves = self.get_valid_king_moves((x,y), player)
                elif piece == PAWN_W:
                    valid_moves = self.get_valid_pawn_moves((x,y), player)
                elif piece == KNIGHT_W:
                    valid_moves = self.get_valid_knight_moves((x,y), player)
                elif piece == BISHOP_W:
                    valid_moves = self.get_valid_bishop_moves((x,y), player)
                elif piece == ROOK_W:
                    valid_moves = self.get_valid_rook_moves((x,y), player)

                from_encoding = (self.width * y + x) * self.width * self.height
                for move_dest in valid_moves:
                    valid[from_encoding + move_dest[1]*self.width+move_dest[0]] = 1




        return valid

    cdef list[tuple[int, int]] get_valid_pawn_moves(self, position : tuple[int, int], player : int):
        cdef list[tuple[int, int]] valid_moves = list()
        if self.pieces[position[0], position[1] + player] == EMPTY:
            valid_moves.append((position[0], position[1] + player))
        if position[0] >= 1 and (self.pieces[position[0] - 1, position[1] + player] - EMPTY) * player > 0:
            valid_moves.append((position[0] - 1, position[1] + player))
        if position[0] < self.width - 1 and (self.pieces[position[0] + 1, position[1] + player] - EMPTY) * player > 0:
            valid_moves.append((position[0] + 1, position[1] + player))

        return valid_moves

    cdef list[tuple[int, int]] get_valid_king_moves(self, position : tuple[int, int], player : int):
        cdef list[tuple[int, int]] valid_moves = list()
        cdef tuple[int, int] dest
        cdef tuple[int, int] move
        for move in KING_MOVES:
            dest = (position[0] + move[0], position[1] + move[1])
            if dest[0] >= self.width or dest[0] < 0 or dest[1] >= self.height or dest[1] < 0:
                continue
            if (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
        return valid_moves


    cdef list[tuple[int, int]] get_valid_knight_moves(self, position : tuple[int, int], player : int):
        cdef list[tuple[int, int]] valid_moves = list()
        cdef tuple[int, int] dest
        cdef tuple[int, int] move
        for move in KNIGHT_MOVES:
            dest = (position[0] + move[0], position[1] + move[1])
            if dest[0] >= self.width or dest[0] < 0 or dest[1] >= self.height or dest[1] < 0:
                continue
            if (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
        return valid_moves

    cdef list[tuple[int, int]] get_valid_rook_moves(self, position : tuple[int, int], player : int):
        cdef list[tuple[int, int]] valid_moves = list()
        cdef tuple[int, int] dest = position
        while(dest[0] >= 1):
            dest = (dest[0] - 1, dest[1])
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        dest = position
        while(dest[0] < self.width - 1):
            dest = (dest[0] + 1, dest[1])
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        dest = position
        while(dest[1] >= 1):
            dest = (dest[0], dest[1] - 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break
        
        dest = position
        while(dest[1] < self.height - 1):
            dest = (dest[0], dest[1] + 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break
        
        return valid_moves

    cdef list[tuple[int, int]] get_valid_bishop_moves(self, position : tuple[int, int], player : int):
        cdef list[tuple[int, int]] valid_moves = list()
        cdef tuple[int, int] dest = position
        while(dest[0] >= 1 and dest[1] >= 1):
            dest = (dest[0] - 1, dest[1] - 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        dest = position
        while(dest[0] < self.width - 1 and dest[1] >= 1):
            dest = (dest[0] + 1, dest[1] - 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        dest = position
        while(dest[0] >= 1 and dest[1] < self.height - 1):
            dest = (dest[0] - 1, dest[1] + 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        dest = position
        while(dest[0] < self.width - 1 and dest[1] < self.height - 1):
            dest = (dest[0] + 1, dest[1] + 1)
            if self.pieces[dest[0],dest[1]] == EMPTY:
                valid_moves.append(dest)
            elif (self.pieces[dest[0],dest[1]] - EMPTY) * player >= 0:
                valid_moves.append(dest)
                break
            else:
                break

        return valid_moves


            

    cpdef tuple[bool,int] get_win_state(self):
        cdef int value
        cdef bint has_white_king = False
        cdef bint has_black_king = False
        cdef Py_ssize_t x, y

        for x in range(self.width):
            for y in range(self.height):
                value = self.pieces[x,y]
                if value == KING_W:
                    has_white_king = True
                elif value == KING_B:
                    has_black_king = True
                
        if not has_white_king:
            return (True, PLAYER_BLACK)
        if not has_black_king:
            return (True, PLAYER_WHITE)
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))
