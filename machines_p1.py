# machines_p1.py

import numpy as np
from mcts_alpha_zero import MCTS
from policy_value_net import PolicyValueNet
import torch

class P1():
    def __init__(self, board, available_pieces, selected_piece=None):
        self.board = board.copy()
        self.available_pieces = available_pieces.copy()
        self.selected_piece = selected_piece

        # 상태 크기와 행동 크기 정의
        self.state_size = 34  # 보드 상태(16) + 남은 말(16) + 선택된 말(1) + 단계(1)
        self.action_size = 16

        # 학습된 모델 로드
        self.policy_value_net = PolicyValueNet(self.state_size, self.action_size)
        self.policy_value_net.load_state_dict(torch.load('best_policy.pth', map_location=torch.device('cpu'), weights_only=True))
        self.policy_value_net.eval()

        # MCTS 초기화
        self.mcts = MCTS(self.policy_value_net, c_puct=5, n_playout=100)

    def select_piece(self, env):
        # 현재 상태 생성
        state = self.get_state(phase='select_piece')
        # MCTS를 사용하여 최적의 행동 선택
        valid_actions = self.get_valid_actions('select_piece')
        action = self.mcts.get_move(env)
        selected_piece = self.available_pieces[action]
        return selected_piece

    def place_piece(self, selected_piece):
        self.selected_piece = selected_piece
        # 현재 상태 생성
        state = self.get_state(phase='place_piece')
        # MCTS를 사용하여 최적의 행동 선택
        valid_actions = self.get_valid_actions('place_piece')
        action = self.mcts.get_move(state, phase='place_piece', valid_actions=valid_actions)
        row, col = divmod(action, 4)
        return (row, col)

    def get_state(self, phase):
        board_state_flattened = self.board.flatten()
        available_pieces_ids = np.array([self.get_piece_id(p)
                                         for p in self.available_pieces])
        if len(available_pieces_ids) < 16:
            available_pieces_ids = np.pad(available_pieces_ids, (0, 16 - len(available_pieces_ids)),
                                          'constant', constant_values=-1)
        selected_piece_id = self.get_piece_id(self.selected_piece) if self.selected_piece else -1
        phase_indicator = 1 if phase == 'select_piece' else 0

        state_vector = np.concatenate([
            board_state_flattened,
            available_pieces_ids,
            np.array([selected_piece_id]),
            np.array([phase_indicator])
        ])
        return state_vector

    def get_valid_actions(self, phase):
        if phase == 'select_piece':
            return [i for i in range(len(self.available_pieces))]
        else:
            return [i for i in range(16) if self.board[i // 4][i % 4] == 0]

    def get_piece_id(self, piece):
        return (piece[0] * 8) + (piece[1] * 4) + (piece[2] * 2) + piece[3]
