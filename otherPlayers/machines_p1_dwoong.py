import numpy as np
import random
from itertools import product
import time
from collections import defaultdict

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        #q테이블 상태값
        self.q_table = defaultdict(float)
        

    # 승리할 수 있는 위치 찾기(바로 승리할 수 있는지)
    def immidiate_win(self, board, piece):
        
        piece_idx = self.pieces.index(piece) + 1

        # 승리 가능한 위치 저장
        winning_pos = []
        
        for row, col in product(range(4), range(4)):
            if board[row][col] == 0:

                #새로운 보드 복사하여 생성 후 값 놓기
                test_board = board.copy()
                test_board[row][col] = piece_idx

                #해당위치가 승리한다면 저장
                if self._check_win(test_board):
                    winning_pos.append((row, col))
                    
        return winning_pos
    
    # 상대 이길 수 있을 때 막기
    def block(self, board, piece):

        piece_idx = self.pieces.index(piece) + 1

        #막을 곳 저장
        block_pos = []
        
        for row, col in product(range(4), range(4)):
            if board[row][col] == 0:

                #상대 이기는 지 파악하기
                dangerous = False #초기에 false로 설정
                test_board = board.copy()
                test_board[row][col] = piece_idx
                
                #available_peices 다 테스트하기
                for test_piece in self.available_pieces:

                    #같지 않을 때만 테스트 왜? 두 번 검사안하도록 하기 위해서
                    if test_piece != piece:
                        for r, c in product(range(4), range(4)):
                            #비어 있는지 확인
                            if test_board[r][c] == 0:
                                test2_board = test_board.copy()
                                test2_board[r][c] = self.pieces.index(test_piece) + 1

                                if self._check_win(test2_board):
                                    #승리한다면 true로 설정
                                    dangerous = True
                                    break
                
                if dangerous:
                    block_pos.append((row, col))
                    
        return block_pos
    
    def evaluate_danger(self, piece):
        
        danger_score = 0
        
        
        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                test_board = self.board.copy()
                test_board[row][col] = self.pieces.index(piece) + 1
                
                
                for test_piece in self.available_pieces:
                    if test_piece != piece:
                        winning = self.immidiate_win(test_board, test_piece)
                        danger_score += len(winning)
        
        return danger_score
    
    #아래는 뭐 승리, 줄, 2x2 확인하기 그냥 짜~
    
    def _check_win(self, board):
        
        for i in range(4):
            row = [board[i][j] for j in range(4)]
            col = [board[j][i] for j in range(4)]
            if self.check_line(row) or self.check_line(col):
                return True
        
        
        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3-i] for i in range(4)]
        if self.check_line(diag1) or self.check_line(diag2):
            return True
        
        
        for i in range(3):
            for j in range(3):
                square = [board[i][j], board[i][j+1], board[i+1][j], board[i+1][j+1]]
                if self._check_square(square):
                    return True
        return False
    
    def check_line(self, line):
        if 0 in line:
            return False
        pieces = [self.pieces[idx-1] for idx in line if idx > 0]
        if len(pieces) != 4:
            return False
        return any(all(p[i] == pieces[0][i] for p in pieces) for i in range(4))
    
    def _check_square(self, square):
        if 0 in square:
            return False
        pieces = [self.pieces[idx-1] for idx in square if idx > 0]
        if len(pieces) != 4:
            return False
        return any(all(p[i] == pieces[0][i] for p in pieces) for i in range(4))
    




    def evaluate_st(self, row, col):
        
        #계산ㄴ
        center_score = 4 - (abs(row-1.5) + abs(col-1.5))
        
        
        adjacent_empty = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc

            #? 계산,,
            if 0 <= new_row < 4 and 0 <= new_col < 4:
                if self.board[new_row][new_col] == 0:
                    adjacent_empty += 1
        
        return center_score + adjacent_empty * 0.5
    


    def select_piece(self):
        if not self.available_pieces:
            return None

        #random 초기화
        random.seed(time.time())

        #랜덤하게 섞기
        shuffled_pieces = self.available_pieces[:]
        random.shuffle(shuffled_pieces)

        
        safe_piece = []
        danger_piece = []

        for piece in shuffled_pieces:
            danger_score = self.evaluate_danger(piece)
            if danger_score == 0:
                safe_piece.append(piece)
            else:
                danger_piece.append((piece, danger_score))

        #safe 값 중 랜덤으로 선택
        if safe_piece:
            best_piece = None
            best_score = float('-inf')

            for piece in safe_piece:
                score = 0

                for row, col in product(range(4), range(4)):
                    if self.board[row][col] == 0:
                        test_board = self.board.copy()
                        test_board[row][col] = self.pieces.index(piece) + 1

                        score += len(self.immidiate_win(test_board, piece))

                if score > best_score:
                    best_score = score
                    best_piece = piece

            #점수 같을 시 랜덤으로 선택
            if best_piece:
                return best_piece
            return random.choice(safe_piece)

        #덜 위험한 것 중 낮은 위험 선택 / 동점일 시 랜덤으로
        if danger_piece:
            danger_min = min(danger_piece, key=lambda x: x[1])[1]
            danger_min_list = [piece for piece, score in danger_piece if score == danger_min]
            return random.choice(danger_min_list)

        #아무것도 아니면 랜덤으로 선택
        return random.choice(shuffled_pieces)



    def place_piece(self, selected_piece):
        available_location = [(row, col) for row, col in product(range(4), range(4))
                                if self.board[row][col] == 0]

        #매번 랜덤으로 하기
        random.seed(time.time())

        #바로 승리 가능한지 위치 확인
        winning = self.immidiate_win(self.board, selected_piece)
        if winning:
            return random.choice(winning)

        #상대방 승리 막기
        blocking_moves = self.block(self.board, selected_piece)
        if blocking_moves:
            best_blocking_move = max(blocking_moves,
                                    key=lambda pos: self.evaluate_st(pos[0], pos[1]))
            return best_blocking_move

        #점수 가장 좋은 곳 찾기
        best_pose = None
        best_score = float('-inf')

        for row, col in available_location:
            score = self.evaluate_st(row, col)

            test_board = self.board.copy()
            test_board[row][col] = self.pieces.index(selected_piece) + 1

            for test_piece in self.available_pieces:
                if test_piece != selected_piece:
                    score -= len(self.immidiate_win(test_board, test_piece)) * 2

            if score > best_score:
                best_score = score
                best_pose = (row, col)

        #만약 위치 점수가 똑같으면 위치 랜덤으로 선택하기
        return best_pose or random.choice(available_location)  