# train.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from policy_value_net import PolicyValueNet
from mcts_alpha_zero import MCTS
from utils import QuartoEnv

def train():
    state_size = 34  # 상태 크기 정의
    action_size = 16  # 행동 크기 정의

    policy_value_net = PolicyValueNet(state_size, action_size)
    optimizer = optim.Adam(policy_value_net.parameters(), lr=0.001)

    num_iterations = 10
    games_per_iteration = 10
    n_playout = 100  # MCTS 시뮬레이션 횟수
    c_puct = 5

    for iteration in range(num_iterations):
        data_buffer = []
        for _ in range(games_per_iteration):
            game_data = self_play(policy_value_net, n_playout, c_puct)
            data_buffer.extend(game_data)

        # 배치 학습
        if len(data_buffer) > 0:
            batch_size = min(len(data_buffer), 64)
            mini_batches = [random.sample(data_buffer, batch_size) for _ in range(1)]
            for batch in mini_batches:
                state_batch = torch.tensor(np.array([data[0] for data in batch]), dtype=torch.float)
                mcts_probs_batch = torch.tensor(np.array([data[1] for data in batch]), dtype=torch.float)
                reward_batch = torch.tensor(np.array([data[2] for data in batch]), dtype=torch.float)

                policy, value = policy_value_net(state_batch)
                value_loss = F.mse_loss(value.view(-1), reward_batch)
                policy_loss = -torch.mean(torch.sum(mcts_probs_batch * F.log_softmax(policy, dim=1), dim=1))
                loss = value_loss + policy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Iteration {iteration+1}, Loss: {loss.item()}")

    # 학습된 모델 저장
    torch.save(policy_value_net.state_dict(), 'best_policy.pth')



def self_play(policy_value_net, n_playout, c_puct):
    env = QuartoEnv()
    state_vector = env.reset()
    data = []
    mcts = MCTS(policy_value_net, c_puct=c_puct, n_playout=n_playout)  # MCTS를 루프 외부로 이동
    while not env.done:
        # 현재 단계와 유효한 행동 가져오기
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break  # 유효한 행동이 없으면 게임 종료
        # MCTS를 사용하여 행동 선택
        try:
            action = mcts.get_move(env)
        except ValueError as e:
            print(f"MCTS 에러: {e}")
            break
        # 선택한 행동이 유효한지 확인
        if action not in valid_actions:
            print(f"Invalid action selected: {action}")
            print(f"Valid actions: {valid_actions}")
            raise ValueError(f"Invalid action selected: {action}")
        # 데이터 저장
        action_probs = np.zeros(16)
        for act, node in mcts.root.children.items():
            action_probs[act] = node.n_visits
        sum_counts = np.sum(action_probs)
        if sum_counts > 0:
            action_probs /= sum_counts
        else:
            action_probs[valid_actions] = 1 / len(valid_actions)
        data.append((state_vector, action_probs, None))
        # 행동 적용
        state_vector, reward, done = env.step(action)
        # MCTS 업데이트
        mcts.update_with_action(action)
    # 게임 결과로 데이터 업데이트
    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1], reward)
    return data





if __name__ == "__main__":
    train()
