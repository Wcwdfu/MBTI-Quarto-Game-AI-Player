# machines_p1.py

import numpy as np
from mcts_alpha_zero import MCTS
from policy_value_net import PolicyValueNet
import torch
from utils import QuartoEnv
from utils import QuartoEnv, get_piece_id, check_if_piece_gives_opponent_win, check_immediate_winning_moves
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class P1():
    def __init__(self, board, available_pieces, selected_piece=None):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.selected_piece = selected_piece

        # 상태 크기와 행동 크기 정의
        self.state_size = 35  # 보드 상태(16) + 남은 말(16) + 선택된 말(1) + 단계(1)
        self.action_size = 16

        # 학습된 모델 로드
        self.policy_value_net = PolicyValueNet(self.state_size, self.action_size).to(device)
        self.policy_value_net.load_state_dict(torch.load('best_policy.pth', map_location=device, weights_only=True))
        self.policy_value_net.eval()

        # MCTS 초기화
        self.mcts = MCTS(self.policy_value_net, c_puct=5, n_playout=100)



    def select_piece(self):
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = self.selected_piece
        env.phase = 'select_piece'
        env.current_player = 1

        # 후보 말 필터링
        safe_pieces = []
        for p in self.available_pieces:
            # 이 말을 넘겼을 때 상대가 즉시 이길 수 있는지 체크
            if not check_if_piece_gives_opponent_win(env.board, p):
                safe_pieces.append(p)

        if len(safe_pieces) == 0:
            # 모두 위험하다면 어쩔 수 없이 MCTS에 맡기거나 랜덤 선택
            # 여기서는 일단 MCTS 돌리게
            action = self.mcts.get_move(env)
        else:
            # safe_pieces 중 임의 선택 또는 추가 전략 가능
            # 일단 safe_pieces가 1개라면 바로 선택
            # 여러개라면 MCTS를 통해 해당 safe_pieces만 고려하도록 state를 조정할 수도 있음
            # 간단히 임의로 선택
            p = random.choice(safe_pieces)
            action = self.get_piece_id(p)

        # 실제 선택된 piece 구하기
        selected_piece = None
        for p in self.available_pieces:
            if self.get_piece_id(p) == action:
                selected_piece = p
                break

        if selected_piece is None:
            raise ValueError(f"Selected action {action} is invalid.")

        # MCTS 업데이트
        self.mcts.update_with_action(action)

        return selected_piece




    def place_piece(self, selected_piece):
        self.selected_piece = selected_piece
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = self.selected_piece
        env.phase = 'place_piece'
        env.current_player = 1

        # 즉시 승리 가능한 위치 체크
        piece_id = self.get_piece_id(selected_piece)
        winning_positions = check_immediate_winning_moves(env.board, piece_id)
        if len(winning_positions) > 0:
            # 즉시 승리 가능한 위치가 있다면 그 중 하나를 선택
            row, col = winning_positions[0]
            action = row * 4 + col
            # MCTS 없이 바로 반환
            return (row, col)

        # 즉시 승리할 수 없다면 MCTS로 결정
        action = self.mcts.get_move(env)
        row, col = divmod(action, 4)
        self.mcts.update_with_action(action)
        return (row, col)




    def get_valid_actions(self, phase):
        if phase == 'select_piece':
            # 말의 ID를 반환
            return [self.get_piece_id(p) for p in self.available_pieces]
        else:
            # 보드의 빈 칸 인덱스 반환
            return [i for i in range(16) if self.board[i // 4][i % 4] == 0]



    def get_piece_id(self, piece):
        return (piece[0] << 3) | (piece[1] << 2) | (piece[2] << 1) | piece[3]
