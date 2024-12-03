# train.py

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from policy_value_net import PolicyValueNet
from mcts_alpha_zero import MCTS
from utils import QuartoEnv
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    state_size = 35  # 상태 크기 정의
    action_size = 16  # 행동 크기 정의

    policy_value_net = PolicyValueNet(state_size, action_size).to(device)
    optimizer = optim.Adam(policy_value_net.parameters(), lr=0.001)


    #--------------------------------------------------------#
    num_iterations = 1
    games_per_iteration = 10
    n_playout = 1000  # MCTS 시뮬레이션 횟수
    c_puct = 5
    #--------------------------------------------------------#


    # 로그 파일 관리 변수
    log_dir = "game_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_index = 1
    games_logged = 0
    max_games_per_file = 100
    log_file = open(os.path.join(log_dir, f"game_logs_{log_file_index}.txt"), "w", encoding="utf-8")

    for iteration in range(num_iterations):
        data_buffer = []
        for game_num in range(games_per_iteration):
            game_data, move_steps = self_play(policy_value_net, n_playout, c_puct)
            data_buffer.extend(game_data)

            # 게임 기록을 로그 파일에 기록
            for step in move_steps:
                log_file.write(step + "\n")
            log_file.write("\n" + "="*40 + " 게임 종료 " + "="*40 + "\n\n")

            games_logged += 1

            # 최대 게임 수 도달 시 새로운 로그 파일 생성
            if games_logged >= max_games_per_file:
                log_file.close()
                log_file_index += 1
                log_file = open(os.path.join(log_dir, f"game_logs_{log_file_index}.txt"), "w", encoding="utf-8")
                games_logged = 0

        # 배치 학습
        if len(data_buffer) > 0:
            batch_size = min(len(data_buffer), 64)
            mini_batches = [random.sample(data_buffer, batch_size) for _ in range(1)]
            for batch in mini_batches:
                state_batch = torch.tensor(np.array([data[0] for data in batch]), dtype=torch.float).to(device)
                mcts_probs_batch = torch.tensor(np.array([data[1] for data in batch]), dtype=torch.float).to(device)
                reward_batch = torch.tensor(np.array([data[2] for data in batch]), dtype=torch.float).to(device)

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
    log_file.close()

def self_play(policy_value_net, n_playout, c_puct):
    env = QuartoEnv()
    state_vector = env.reset()
    data = []
    move_steps = []  # 게임 기록을 저장할 리스트
    mcts = MCTS(policy_value_net, c_puct=c_puct, n_playout=n_playout)

    while not env.done:
        current_player = env.current_player
        player_name = f"P{current_player}"
        if current_player == 1:
            mbti_type = "P1"  # 또는 실제 MBTI 타입을 매핑
        else:
            mbti_type = "P2"  # 또는 실제 MBTI 타입을 매핑

        # 현재 단계에 따라 행동 선택
        try:
            action = mcts.get_move(env)
        except ValueError as e:
            print(f"MCTS 에러: {e}")
            break

        valid_actions = env.get_valid_actions()
        if action not in valid_actions:
            print(f"Invalid action selected: {action}")
            print(f"Valid actions: {valid_actions}")
            raise ValueError(f"Invalid action selected: {action}")

        # 현재 단계가 'select_piece'인지 'place_piece'인지 확인
        if env.phase == 'select_piece':
            # 선택된 말의 MBTI 타입을 가져오기
            selected_piece = env.available_pieces[[env.get_piece_id(p) for p in env.available_pieces].index(action)]
            mbti_selected = piece_to_mbti(selected_piece)
            move_steps.append(f"{player_name}가 {mbti_selected} 선택")
        else:
            # 배치할 위치를 가져오기
            row, col = divmod(action, 4)
            move_steps.append(f"{player_name}가 ({row},{col})에 둠")

        # 데이터 저장
        action_probs = np.zeros(16)
        for act, node in mcts.root.children.items():
            action_probs[act] = node.n_visits
        sum_counts = np.sum(action_probs)
        if sum_counts > 0:
            action_probs /= sum_counts
        else:
            action_probs[valid_actions] = 1 / len(valid_actions)
        # 현재 플레이어 정보 추가하여 저장
        data.append((state_vector, action_probs, None, current_player))

        # 행동 적용
        state_vector, reward, done = env.step(action)
        # MCTS 업데이트
        mcts.update_with_action(action)
        # 현재 상태 벡터를 다음 반복에서 사용하기 위해 저장
        state_vector = env.get_state_vector()

    # 게임 결과에 따라 승리/패배/무승부 기록
    if env.winner == 1:
        move_steps.append("P1 승리")
        game_result = 1
    elif env.winner == 2:
        move_steps.append("P2 승리")
        game_result = -1
    else:
        move_steps.append("무승부")
        game_result = 0

    # 게임 데이터의 reward 업데이트
    for i in range(len(data)):
        # 데이터의 현재 플레이어 정보를 사용하여 보상을 설정
        current_player = data[i][3]  # 저장된 현재 플레이어
        reward = game_result if current_player == 1 else -game_result
        data[i] = (data[i][0], data[i][1], reward)

    return data, move_steps

def piece_to_mbti(piece):
    # MBTI 타입을 문자열로 변환하는 함수
    # MBTI Pieces (Binary Encoding: I/E = 0/1, N/S = 0/1, T/F = 0/1, P/J = 0/1)
    mapping = {
        (0,0,0,0): "INTP",
        (0,0,0,1): "INTJ",
        (0,0,1,0): "INFJ",
        (0,0,1,1): "INFP",
        (0,1,0,0): "ISTP",
        (0,1,0,1): "ISTJ",
        (0,1,1,0): "ISFP",
        (0,1,1,1): "ISFJ",
        (1,0,0,0): "ENTP",
        (1,0,0,1): "ENTJ",
        (1,0,1,0): "ENFJ",
        (1,0,1,1): "ENFP",
        (1,1,0,0): "ESTP",
        (1,1,0,1): "ESTJ",
        (1,1,1,0): "ESFP",
        (1,1,1,1): "ESFJ",
    }
    return mapping.get(piece, "UNKNOWN")

if __name__ == "__main__":
    train()
