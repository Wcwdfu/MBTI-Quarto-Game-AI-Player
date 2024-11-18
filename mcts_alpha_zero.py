# mcts_alpha_zero.py

import numpy as np
import copy
import math
import torch

class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # action -> TreeNode
        self.n_visits = 0
        self.Q = 0
        self.U = 0
        self.P = prior_p  # prior probability

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # 누적된 Q값 업데이트
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        # 부모 노드까지 역전파
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        # UCB 값 계산
        self.U = (c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.U

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=100):
        self.policy_value_fn = policy_value_fn  # 상태에서 (action_probs, value)를 반환하는 함수
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = TreeNode(None, 1.0)

    def _playout(self, state):
        node = self.root
        while True:
            if node.is_leaf():
                break
            # 노드 선택
            action, node = node.select(self.c_puct)
            # 선택한 행동이 유효한지 확인
            if action not in state.get_valid_actions():
                # 유효하지 않은 행동이면 노드를 삭제하고 다시 선택
                del node.parent.children[action]
                return
            # 행동 적용
            state.do_move(action, state.phase)
            if state.done:
                # 게임이 종료된 상태에서는 더 이상 플레이아웃을 진행하지 않음
                return
        # 리프 노드 평가
        if state.done:
            # 게임이 종료된 상태에서는 leaf_value를 결정
            if state.winner == 1:
                leaf_value = 1
            elif state.winner == 2:
                leaf_value = -1
            else:
                leaf_value = 0
        else:
            state_vector = state.get_state_vector()
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_logits, leaf_value = self.policy_value_fn(state_tensor)
            action_probs = torch.softmax(action_logits, dim=1).detach().numpy()[0]
            # 유효한 행동만 고려
            valid_actions = state.get_valid_actions()
            action_probs_masked = np.zeros_like(action_probs)
            action_probs_masked[valid_actions] = action_probs[valid_actions]
            sum_probs = np.sum(action_probs_masked)
            if sum_probs > 0:
                action_probs_masked /= sum_probs
            else:
                # 유효한 행동이 없을 경우, 플레이아웃을 종료
                return
            # 노드 확장
            action_priors = [(a, p) for a, p in enumerate(action_probs_masked) if p > 0]
            node.expand(action_priors)
        # 역전파
        node.update_recursive(-leaf_value)

    def get_move(self, state):
        if state.done:
            raise ValueError("Cannot get move from a terminal state.")
        for _ in range(self.n_playout):
            state_copy = state.copy()
            self._playout(state_copy)
        # 방문 횟수를 기반으로 행동 선택
        visit_counts = np.zeros(16)
        for action, child in self.root.children.items():
            visit_counts[action] = child.n_visits
        # 유효한 행동만 고려하여 확률 계산
        valid_actions = state.get_valid_actions()
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")
        action_probs = np.zeros(16)
        action_probs[valid_actions] = visit_counts[valid_actions]
        sum_counts = np.sum(action_probs)
        if sum_counts > 0:
            action_probs /= sum_counts
        else:
            action_probs[valid_actions] = 1 / len(valid_actions)
        # 행동 선택
        action = np.random.choice(valid_actions, p=action_probs[valid_actions])
        return action

    def update_with_action(self, last_action):
        if last_action in self.root.children:
            self.root = self.root.children[last_action]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
