# machines_p1.py

import numpy as np
from mcts_alpha_zero import MCTS
from policy_value_net import PolicyValueNet
import torch
from utils import QuartoEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class P2():
    def __init__(self, board, available_pieces, selected_piece=None):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.selected_piece = selected_piece

        # 상태 크기와 행동 크기 정의
        self.state_size = 34  # 보드 상태(16) + 남은 말(16) + 선택된 말(1) + 단계(1)
        self.action_size = 16

        # 학습된 모델 로드
        self.policy_value_net = PolicyValueNet(self.state_size, self.action_size).to(device)
        self.policy_value_net.load_state_dict(torch.load('best_policy.pth', map_location=device, weights_only=True))
        self.policy_value_net.eval()

        # MCTS 초기화
        self.mcts = MCTS(self.policy_value_net, c_puct=5, n_playout=100)

    def select_piece(self):
        # QuartoEnv 객체 생성 및 현재 상태로 초기화
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = self.selected_piece
        env.phase = 'select_piece'
        env.current_player = 1  # P1이 플레이어인 경우

        # MCTS를 사용하여 최적의 행동 선택
        action = self.mcts.get_move(env)  # action은 말의 ID

        # 선택된 action이 유효한지 확인하고 해당 말을 찾음
        selected_piece = None
        for p in self.available_pieces:
            if self.get_piece_id(p) == action:
                selected_piece = p
                break
        if selected_piece is None:
            raise ValueError(f"Selected action {action} is out of bounds for available_pieces.")

        # MCTS의 루트 노드 업데이트
        self.mcts.update_with_action(action)

        return selected_piece

    def place_piece(self, selected_piece):
        self.selected_piece = selected_piece

        # QuartoEnv 객체 생성 및 현재 상태로 초기화
        env = QuartoEnv()
        env.board = self.board.copy()
        env.available_pieces = self.available_pieces.copy()
        env.selected_piece = self.selected_piece
        env.phase = 'place_piece'
        env.current_player = 1  # P1이 플레이어인 경우

        # MCTS를 사용하여 최적의 행동 선택
        action = self.mcts.get_move(env)  # action은 배치할 위치 (0-15)

        row, col = divmod(action, 4)

        # MCTS의 루트 노드 업데이트
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
