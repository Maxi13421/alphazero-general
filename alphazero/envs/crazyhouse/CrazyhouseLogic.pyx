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


    cdef int width
    cdef int height
    cdef int placable_pieces

    def __init__(self, width, height, placable_pieces):
        """Set up initial board configuration."""

        self.width = width
        self.height = height
        self.placable_pieces = placable_pieces

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

        self.piece_counts_white = np.zeros(placable_pieces, dtype=np.intc)
        self.piece_counts_black = np.zeros(placable_pieces, dtype=np.intc)

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

    #If old_y == self.height: old_x + 1 is piece to place    
    #Only works if num_pieces (without king) <= self.width
    #Currently wasted actions because of pawns can't be placed on first and last row

    cpdef int[:] get_move(self, action : int):
        cdef int[:] move = np.empty(4, dtype=np.intc)  
        cdef int remainder
        remainder, move[2] = divmod(action, self.width)
        remainder, move[3] = divmod(remainder, self.height)
        move[1], move[0] = divmod(remainder, self.width)
        return move

    cpdef void move(self, int action):
        cdef int[:] move = self.get_move(action)
        #print("{} {} {} {}".format(move[0], move[1], move[2], move[3]))
        cdef int piece
        cdef int piece_on_dest
        if(move[1] < self.height):
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
            if move[0] == self.width - 1 and move[1] == 0:
                self.castling_rights[1] = False  

            if piece == KING_W and move[0] - move[2] == 2:
                self.pieces[0,0] = EMPTY
                self.pieces[3,0] = ROOK_W
            if piece == KING_W and move[2] - move[0] == 2:
                self.pieces[7,0] = EMPTY
                self.pieces[5,0] = ROOK_W

            if piece == PAWN_W and move[1] == self.height - 4 and self.pieces[move[2], move[1]] == PAWN_B and \
                    (move[0] == move[2] + 1 or move[0] == move[2] - 1) and self.last_action != -1: #En passant
                last_move = self.get_move(self.last_action)
                if last_move[0] == move[2] and self.height - 1 - last_move[1] == move[1] + 2 \
                        and last_move[2] == move[2] and self.height - 1 - last_move[3] == move[1]:
                    self.piece_counts_white[0] += 1
                    self.pieces[move[2], move[1]] = EMPTY

            if self.is_piece_promoted[move[0], move[1]]:
                self.is_piece_promoted[move[2], move[3]] = True
                self.is_piece_promoted[move[0], move[1]] = False

            if piece == PAWN_W and move[3] == self.height - 1:
                self.pieces[move[2], move[3]] = QUEEN_W
                self.is_piece_promoted[move[2], move[3]] = True
            else:   
                self.pieces[move[2], move[3]] = piece
            self.pieces[move[0], move[1]] = EMPTY
        else:
            self.pieces[move[2], move[3]] = move[0] + 1
            self.piece_counts_white[move[0]] -= 1

        cdef Py_ssize_t x, y
        for y in range(self.height // 2):
            for x in range(self.width):
                self.pieces[x, y], self.pieces[x, self.height - 1 - y] = KING_B - self.pieces[x, self.height - 1 - y], KING_B - self.pieces[x, y]
                self.is_piece_promoted[x, y], self.is_piece_promoted[x, self.height - 1 - y] = self.is_piece_promoted[x, self.height - 1 - y], self.is_piece_promoted[x, y]
        if (self.height & 1) == 1:
            for x in range(self.width):
                self.pieces[x, self.height // 2] = KING_B - self.pieces[x, self.height // 2]

        self.last_action = action

        self.piece_counts_white, self.piece_counts_black = self.piece_counts_black, self.piece_counts_white

        self.castling_rights[0], self.castling_rights[0] = self.castling_rights[2], self.castling_rights[0]
        self.castling_rights[1], self.castling_rights[3] = self.castling_rights[3], self.castling_rights[1]



    cpdef list[int[2]] legal_moves(self, int start_x, int start_y):
        if(start_y == self.height):
            return self.get_valid_place_moves(start_x)
        piece = self.pieces[start_x,start_y]
        if piece - EMPTY >= 0:
            return list() 

        if piece == KING_W:
            return self.get_valid_king_moves(start_x, start_y)
        elif piece == PAWN_W:
            return self.get_valid_pawn_moves(start_x, start_y)
        elif piece == KNIGHT_W:
            return self.get_valid_knight_moves(start_x, start_y)
        elif piece == BISHOP_W:
            return self.get_valid_bishop_moves(start_x, start_y)
        elif piece == ROOK_W:
            return self.get_valid_rook_moves(start_x, start_y)
        elif piece == QUEEN_W:
            return self.get_valid_queen_moves(start_x, start_y)


    cpdef int[:] get_valid_moves(self):
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros(((self.width*self.height + self.placable_pieces)*self.width*self.height), dtype=np.intc)
        cdef list[int[2]] valid_moves
        cdef int from_encoding, index

        cdef Py_ssize_t x, y, p
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
                elif piece == QUEEN_W:
                    valid_moves = self.get_valid_queen_moves(x,y)

                from_encoding = (self.width * y + x) * self.width * self.height
                for move_dest in valid_moves:
                    index = from_encoding + move_dest[1]*self.width+move_dest[0]
                    valid[index] = 1

        for p in range(self.placable_pieces):
            valid_moves = self.get_valid_place_moves(p)
            from_encoding = (self.width * self.height + p) * self.width * self.height
            for move_dest in valid_moves:
                index = from_encoding + move_dest[1]*self.width+move_dest[0]
                valid[index] = 1


        return valid

    cdef list[int[2]] get_valid_place_moves(self, int piece_type_minus_one):
        cdef list[int[2]] valid_moves = list()
        if(self.piece_counts_white[piece_type_minus_one] == 0): return valid_moves
        cdef Py_ssize_t x, y
        if piece_type_minus_one == 0:
            for y in range(1, self.height - 1):
                for x in range(self.width):
                    if(self.pieces[x,y] == EMPTY):
                        valid_moves.append([x,y])   
        else:
            for y in range(self.height):
                for x in range(self.width):
                    if(self.pieces[x,y] == EMPTY):
                        valid_moves.append([x,y])
        return valid_moves
        

    cdef list[int[2]] get_valid_pawn_moves(self, int position_x, int position_y):
        cdef list[int[2]] valid_moves = list()
        if self.pieces[position_x, position_y + 1] == EMPTY:
            valid_moves.append([position_x, position_y + 1])
            if position_y == 1 and self.pieces[position_x, 3] == EMPTY:
                valid_moves.append([position_x, 3])
        if position_x >= 1 and (self.pieces[position_x - 1, position_y + 1] - EMPTY) > 0:
            valid_moves.append([position_x - 1, position_y + 1])
        if position_x < self.width - 1 and (self.pieces[position_x + 1, position_y + 1] - EMPTY) > 0:
            valid_moves.append([position_x + 1, position_y + 1])

        #En passant
        if position_y == self.height - 4 and position_x >= 1 and self.pieces[position_x - 1, position_y] == PAWN_B and \
                    self.last_action != -1: 
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x - 1 and self.height - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x - 1 and self.height - 1 - last_move[3] == position_y:
                valid_moves.append([position_x - 1, position_y + 1])
        if position_y == self.height - 4 and position_x < self.width - 1 and self.pieces[position_x + 1, position_y] == PAWN_B and \
                    self.last_action != -1:
            last_move = self.get_move(self.last_action)
            if last_move[0] == position_x + 1 and self.height - 1 - last_move[1] == position_y + 2 \
                    and last_move[2] == position_x + 1 and self.height - 1 - last_move[3] == position_y:
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

        if self.castling_rights[0] and self.pieces[1,0] == EMPTY and self.pieces[2,0] == EMPTY and self.pieces[3,0] == EMPTY \
            and not self.square_is_attacked_by_black(3,0) and not self.square_is_attacked_by_black(4,0):
            valid_moves.append([2,0])

        if self.castling_rights[1] and self.pieces[5,0] == EMPTY and self.pieces[6,0] == EMPTY \
            and not self.square_is_attacked_by_black(4,0) and not self.square_is_attacked_by_black(5,0):
            valid_moves.append([6,0])

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

    cdef list[int[2]] get_valid_queen_moves(self, int position_x, int position_y):
        return self.get_valid_rook_moves(position_x, position_y) + self.get_valid_bishop_moves(position_x, position_y)


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


    cdef int square_is_attacked_by_black(self, int position_x, int position_y):
        return self.square_is_attacked_by_black_pawn(position_x, position_y) or self.square_is_attacked_by_black_rook_or_queen(position_x, position_y) or \
            self.square_is_attacked_by_black_bishop_or_queen(position_x, position_y) or self.square_is_attacked_by_black_king(position_x, position_y)
    
    #Not really necessary
    cdef int square_is_attacked_by_black_king(self, int position_x, int position_y):
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KING_MOVES[move]
            dest_y = position_y + KING_MOVES[move+1]
            if dest_x >= self.width or dest_x < 0 or dest_y >= self.height or dest_y < 0:
                continue
            if self.pieces[dest_x,dest_y] == KING_B:
                return True
        return False

    cdef int square_is_attacked_by_black_pawn(self, int position_x, int position_y):
        if position_x >= 1 and self.pieces[position_x - 1, position_y + 1] == PAWN_B:
            return True
        if position_x < self.width - 1 and self.pieces[position_x + 1, position_y + 1] == PAWN_B:
            return True

    cdef int square_is_attacked_by_black_knight(self, int position_x, int position_y):
        cdef int dest_x, dest_y
        cdef Py_ssize_t move
        for move in range(0,16,2):
            dest_x = position_x + KNIGHT_MOVES[move]
            dest_y = position_y + KNIGHT_MOVES[move+1]
            if dest_x >= self.width or dest_x < 0 or dest_y >= self.height or dest_y < 0:
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
        while(dest_x < self.width - 1):
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
        while(dest_y < self.height - 1):
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
        while(dest_x < self.width - 1 and dest_y >= 1):
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
        while(dest_x >= 1 and dest_y < self.height - 1):
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
        while(dest_x < self.width - 1 and dest_y < self.height - 1):
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

        for x in range(self.width):
            for y in range(self.height):
                value = self.pieces[x,y]
                if value == KING_W:
                    has_white_king = True
                
        if not has_white_king:
            return (True, -player)
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.piece_counts_black)) + "\n" + str(np.asarray(self.pieces)) \
             + "\n" + str(np.asarray(self.piece_counts_white)) + "\n" + "\n" + str(np.asarray(self.is_piece_promoted)) \
             + "\n" + str(np.asarray(self.castling_rights)) + "\n" + str(np.asarray(self.last_action))
