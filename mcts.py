import math
import random
import numpy as np
from copy import deepcopy

from rope.base.oi.type_hinting.evaluate import evaluate


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.action = action  # 현재 노드의 행동 (말 선택 또는 위치 선택)

    def is_fully_expanded(self):
        if self.state.current_piece is None:  # 말 선택 단계
            return len(self.children) == len(self.state.remaining_pieces)
        else:  # 위치 선택 단계
            return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            raise ValueError("No children available to select the best child.")
        choices_weights = []
        for child in self.children:
            if child.visits > 0:
                exploitation = child.value / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(exploitation + exploration)
            else:
                choices_weights.append(float('inf'))  # 방문하지 않은 자식 노드를 우선 탐색
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, itermax=1000, max_depth=8):
        """
        MCTS 탐색 실행.
        - 최대 반복(itermax)과 깊이 제한(max_depth)을 설정.
        """
        root = MCTSNode(state=initial_state)
        for _ in range(itermax):
            node = self._select(root)
            if not node.state.is_terminal():
                reward = self._simulate(node.state, max_depth=max_depth)
                self._backpropagate(node, reward)
            else:
                self._backpropagate(node, node.state.get_result())

        if not root.children:
            raise ValueError("Search completed but no children were expanded. Check expand logic.")

        # 최종 best_child 선택 디버깅 메시지 출력
        print("==== Final Best Child Selection ====")
        for child in root.children:
            print(f"Action: {child.action}, Visits: {child.visits}, Value: {child.value}")

        best_child = root.best_child(0)  # exploration_weight=0: 순수 exploitation
        print(
            f"Selected Best Child: Action: {best_child.action}, Visits: {best_child.visits}, Value: {best_child.value}")
        print("====================================")
        return best_child.state

    def _select(self, node):
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        if not node.state.is_terminal():
            return self._expand(node)
        return node

    def _expand(self, node):
        if node.state.current_piece is None:  # 말 선택 단계
            tried_pieces = [child.action for child in node.children]
            for piece in node.state.remaining_pieces:
                if piece not in tried_pieces:
                    new_state = deepcopy(node.state)
                    new_state.make_move(piece)
                    child_node = MCTSNode(state=new_state, parent=node, action=piece)
                    node.children.append(child_node)
                    return child_node
        else:  # 위치 선택 단계
            tried_moves = [child.action for child in node.children]
            for move in node.state.get_legal_moves():
                if move not in tried_moves:
                    new_state = deepcopy(node.state)
                    new_state.make_move(move)
                    child_node = MCTSNode(state=new_state, parent=node, action=move)
                    node.children.append(child_node)
                    return child_node
        return None

    def _simulate(self, state, max_depth=8):
        current_state = deepcopy(state)
        depth = 0
        while not current_state.is_terminal() and depth < max_depth:
            if current_state.current_piece is None:  # 말 선택 단계
                piece = random.choice(current_state.remaining_pieces)
                current_state.make_move(piece)
            else:  # 위치 선택 단계
                move = random.choice(current_state.get_legal_moves())
                current_state.make_move(move)
            depth += 1

        # 터미널 상태에서 결과 반환
        if current_state.is_terminal():
            return current_state.get_result()
        else:
            # 최대 깊이에 도달하면 휴리스틱 평가
            return self.heuristic_evaluation(current_state)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            if node.state.inturn == 0:
                if node.state.turn in [0,1]:
                    node.value += reward
                else:
                    node.value -= reward
            else:
                if node.state.turn in [2,3]:
                    node.value += reward
                else:
                    node.value -= reward
            """
            print(f"Backpropagating Node - Action: {node.action}, Turn: {node.state.turn}")
            print(f"Updated Visits: {node.visits}, Value: {node.value}")
            """

            node = node.parent

    def heuristic_evaluation(self, state):
        """
        현재 게임 상태를 평가하여 점수를 반환.
        높은 점수가 더 유리한 상태를 의미.
        """
        score = 0

        # 1. 상대의 승리 방어
        score -= self.evaluate_opponent_threat(state)

        return score

    def evaluate_opponent_threat(self, state):
        """
        상대방의 승리 가능성을 평가.
        - 상대방이 다음 턴에 승리할 가능성이 높은 줄(빈 칸 1개)을 방지.
        """
        lines = []

        # 가로, 세로 줄 추가
        for i in range(4):
            lines.append(state.board[i, :])  # 가로 줄
            lines.append(state.board[:, i])  # 세로 줄

        # 대각선 줄 추가
        lines.append(np.diag(state.board))
        lines.append(np.diag(np.fliplr(state.board)))

        # 2x2 블록 추가
        for i in range(3):
            for j in range(3):
                lines.append([
                    state.board[i, j], state.board[i, j + 1],
                    state.board[i + 1, j], state.board[i + 1, j + 1]
                ])

        score = 0

        for line in lines:
            empty_slots = sum(1 for cell in line if cell == 0)
            if empty_slots == 1:  # 상대방이 완성 가능
                score -= 0.2  # 위협 줄 감소
        return score




class QuartoState:
    def __init__(self, board=None, remaining_pieces=None, current_piece=None, turn=0, inturn = 0):
        # 4x4 numpy 배열로 보드 초기화 (0으로 초기화)
        self.board = board if board is not None else np.zeros((4, 4), dtype=int)
        # 전체 말들 (0과 1로 이루어진 4개의 요소를 가진 튜플 배열)
        self.all_pieces = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
        # 남은 말들 (초기에는 모든 말을 포함)
        self.remaining_pieces = remaining_pieces if remaining_pieces else self.all_pieces[:]
        # 현재 차례에 놓을 말
        self.current_piece = current_piece
        # 현재 턴 (0: 내가 말 선택, 1: 상대가 위치 선택, 2: 상대가 말 선택, 3: 내가 위치 선택)
        self.turn = turn
        # 마지막 움직임
        self.last_move = None
        # 마지막 말 선택
        self.last_piece = None

        self.inturn = inturn

    def get_legal_moves(self):
        # 비어있는 칸의 좌표 반환 (값이 0인 칸)
        return [(row, col) for row, col in zip(*np.where(self.board == 0))]

    def make_move(self, action):
        if self.turn in [0, 2]:  # 말 선택 단계
            if action not in self.remaining_pieces:
                raise ValueError(f"Invalid action: {action} is not available.")
            self.last_piece = action
            self.current_piece = action
        elif self.turn in [1, 3]:  # 위치 선택 단계
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError(f"Invalid action: {action} must be a tuple (row, col).")
            row, col = action
            if self.board[row, col] != 0:
                raise ValueError(f"Illegal move: Position already occupied at {action}")
            piece_index = self.all_pieces.index(self.current_piece) + 1
            self.board[row, col] = piece_index
            self.remaining_pieces.remove(self.current_piece)
            self.last_move = action
            self.current_piece = None  # 다음 단계는 말 선택
        else:
            raise ValueError(f"Invalid turn: {self.turn}")

        # 턴 전환 (0 -> 1 -> 2 -> 3 -> 0)
        self.turn = (self.turn + 1) % 4

    def is_terminal(self):
        # 승리 조건 확인 또는 남은 말이 없는 경우 종료
        if self.check_victory():
            return True
        if not self.remaining_pieces and not self.get_legal_moves():
            return True
        return False

    def get_result(self):
        # 게임 결과 반환 (1: 승리, 0: 무승부, -1: 패배)
        if self.check_victory():
            if self.inturn == 0:
                return 1 if self.turn == 0 else -1
            elif self.inturn == 1:
                return 1 if self.turn == 2 else -1
        elif self.current_piece is None:
            return 0  # 모든 말을 사용했지만 승리 조건이 없음
        return -3

    def check_victory(self):
        # 가로, 세로, 대각선에서 승리 조건 확인
        for row in self.board:
            if self.is_quarto(row):
                return True
        for col in self.board.T:
            if self.is_quarto(col):
                return True
        if self.is_quarto(np.diag(self.board)) or self.is_quarto(np.diag(np.fliplr(self.board))):
            return True

        # 2x2 블록에서 승리 조건 확인
        for i in range(3):  # 0, 1, 2 (블록 시작 행)
            for j in range(3):  # 0, 1, 2 (블록 시작 열)
                block = [
                    self.board[i, j],
                    self.board[i, j + 1],
                    self.board[i + 1, j],
                    self.board[i + 1, j + 1]
                ]
                if self.is_quarto(block):
                    return True
        return False

    def is_quarto(self, line):
        # 4개의 말이 모두 채워져 있고, 공통 속성을 가진 경우
        if 0 in line:  # 값이 0인 경우 비어 있는 칸
            return False
        # 인덱스를 통해 all_pieces에서 속성을 가져옴
        properties = [self.all_pieces[int(piece) - 1] for piece in line]
        return any(all(prop[i] for prop in properties) or not any(prop[i] for prop in properties) for i in range(4))

    def get_all_lines(board):
        """
        보드에서 가로, 세로, 대각선, 2x2 블록을 모두 반환.
        """
        lines = []

        # 가로, 세로 줄 추가
        for i in range(4):
            lines.append(board[i, :])  # 가로 줄
            lines.append(board[:, i])  # 세로 줄

        # 대각선 줄 추가
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))

        # 2x2 블록 추가
        for i in range(3):
            for j in range(3):
                lines.append([
                    board[i, j], board[i, j + 1],
                    board[i + 1, j], board[i + 1, j + 1]
                ])