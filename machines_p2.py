import numpy as np
import random
from itertools import product

import time
import math
import copy

class MCTSNode:
    def __init__(self, board, available_pieces, parent=None, move=None):
        self.board = board
        self.available_pieces = available_pieces
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0


class P2:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.simulation_time = 0.5  # Default : 5 AND Max : 9
        self.exploration_constant = 0.75
        self.move_count = 0
        self.turn = 0

    def select_piece(self):
        root = MCTSNode(self.board, self.available_pieces)
        self.turn = 0

        end_time = time.time() + self.simulation_time

        while time.time() < end_time:
            self.move_count = 0  # Manage all process count

            node = self.select(root, None)
            value = self.simulate(node)
            # value = 1 - self.simulate(node)
            self.backpropagate(node, value)

        best_child = max(root.children, key=lambda c: c.value)
        return best_child.move[0]

    def place_piece(self, selected_piece):
        root = MCTSNode(self.board, self.available_pieces)
        self.turn = 1
        #print(selected_piece)  # test_page

        end_time = time.time() + self.simulation_time

        while time.time() < end_time:
            self.move_count = 0  # Manage all process count

            node = self.select(root, selected_piece)
            value = self.simulate(node)
            self.backpropagate(node, value)

        best_child = max(root.children, key=lambda c: c.visits)
        # print("root's value: " + str(root.value))  # test_page

        return best_child.move[1]

    def select(self, node, piece):
        if piece is None:
            
            safe_piece_list = self.safe_pieces()
            print("safe")
            print(safe_piece_list)
            tttttt = self.available_pieces
            # available_pieces에 safe_piece_list의 항목만 남기기
            self.available_pieces = [piece for piece in self.available_pieces if piece in safe_piece_list]
            print("available")
            print(self.available_pieces)
            if not self.available_pieces:  # available_pieces가 비어 있는 경우 처리
                return random.choice(tttttt)
            
            piece = random.choice(self.available_pieces)  # 안전한 조각 중 선택

            

        while node.children:
            # print("children exist")  # test_pages
            '''
            # Additional child node extensions
            if not all(child.visits > 0 for child in node.children):
                print("[Message] children exist but not visits")  # test_pages
                return self.expand(node, piece)
            '''
            node = self.uct_select(node)
            #return node  # test_page
            # break  # test_page

        #print("[Message] no children")  # test_page
        self.move_count += 1
        return self.expand(node, piece)

    def expand(self, node, piece):
        if self.is_terminal(node.board):
            return node

        empty_cells = [(r, c) for r, c in product(range(4), repeat=2) if node.board[r][c] == 0]

        for row, col in empty_cells:
            new_board = copy.deepcopy(node.board)

            new_board[row][col] = self.pieces.index(piece) + 1
            new_available_pieces = node.available_pieces[:]

            # print(f"add children in row: {row} col: {col}")  # test_page
            child = MCTSNode(new_board, new_available_pieces, node, (piece, (row, col)))
            node.children.append(child)

        return self.uct_select(node)
        # return random.choice(node.children)

    def simulate(self, node):
        board = copy.deepcopy(node.board)
        
        #print(node.available_pieces)
        available_pieces = node.available_pieces[:]

        while not self.is_terminal(board) and available_pieces:
            piece = random.choice(available_pieces)
            empty_cells = [(r, c) for r, c in product(range(4), repeat=2) if board[r][c] == 0]

            if not empty_cells:
                break

            row, col = random.choice(empty_cells)
            board[row][col] = self.pieces.index(piece) + 1
            available_pieces.remove(piece)
            self.move_count += 1

        return self.evaluate(self.move_count)

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            #print(f"node's visit: {node.visits:>5}")  # test_page
            node.value += value
            #print(f"node's value: {node.value:>5}")  # test_page
            value = 1 - value
            node = node.parent
        #print("--")

    def uct_select(self, node):
        for child in node.children:
            if child.visits == 0:
                return child

        return max(node.children, key=lambda c: c.value / c.visits + 2 * self.exploration_constant * math.sqrt(
                                                math.log(node.visits) / c.visits))

    def safe_pieces(self):
        safe_pieces = []

        for piece in self.available_pieces:
            is_safe = True

            for row, col in product(range(4), repeat=2):
                if self.board[row][col] == 0:
                    temp_board = copy.deepcopy(self.board)
                    temp_board[row][col] = self.pieces.index(piece) + 1

                    if self.check_win(temp_board):
                        is_safe = False
                        break

            if is_safe:
                safe_pieces.append(piece)

        # if safe_pieces:
        #     safe_pieces.append(random.choice(self.available_pieces))


        if not safe_pieces:  # 안전한 조각이 없는 경우
            if self.available_pieces:  # 사용 가능한 조각이 있다면 랜덤으로 반환
                print("난죽택")
                safe_pieces.append(random.choice(self.available_pieces))

        return safe_pieces

    def is_terminal(self, board):
        return self.check_win(board) or all(board[r][c] != 0 for r, c in product(range(4), repeat=2))

    def check_line(self, line):
        if 0 in line:
            return False  # Incomplete line

        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])

        for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
            if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
                return True

        return False

    def check_2x2_subgrid_win(self, board):
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]

                if 0 not in subgrid:  # All cells must be filled
                    characteristics = [self.pieces[idx - 1] for idx in subgrid]

                    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
                        if len(set(char[i] for char in characteristics)) == 1:  # All share the same characteristic
                            return True

        return False

    def check_win(self, board):
        # Check rows, columns, and diagonals
        for col in range(4):
            if self.check_line([board[row][col] for row in range(4)]):
                return True

        for row in range(4):
            if self.check_line([board[row][col] for col in range(4)]):
                return True

        if self.check_line([board[i][i] for i in range(4)]) or self.check_line([board[i][3 - i] for i in range(4)]):
            return True

        # Check 2x2 sub-grids
        if self.check_2x2_subgrid_win(board):
            return True

        return False

    def evaluate(self, count):
        if self.turn == 1 and count % 2 == 1:
            return 1  # 승
        elif self.turn == 0 and count % 2 == 0:
            return 1  # 승
        else:
            return 0  # 패