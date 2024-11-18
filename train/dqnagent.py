import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os  # 파일 및 디렉토리 처리를 위해 추가

class DQNAgent(nn.Module):
    def __init__(self, state_size=34, action_size_select=16, action_size_place=16):
        super(DQNAgent, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.select_head = nn.Linear(64, action_size_select)
        self.place_head = nn.Linear(64, action_size_place)

    def forward(self, x, phase):
        x = self.shared_fc(x)
        if phase == 'select_piece':
            return self.select_head(x)
        else:
            return self.place_head(x)



def train_dqn_with_replay(env, agent, episodes=1000, gamma=0.99, batch_size=32,
                          replay_buffer_size=10000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=replay_buffer_size)
    epsilon = epsilon_start

    # 로그를 저장할 디렉토리 생성
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 에피소드 로그를 저장할 리스트 초기화
    all_logs = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_loss = 0
        step_count = 0

        # 현재 에피소드 로그를 저장할 리스트 초기화
        episode_log = [f"Episode {episode+1}"]

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            phase_indicator = state[-1]
            phase = 'select_piece' if phase_indicator == 1 else 'place_piece'

            # 유효한 행동 리스트 생성
            if phase == 'select_piece':
                valid_actions = [i for i in range(16) if env.all_pieces[i] in env.available_pieces]
            else:
                valid_actions = [i for i in range(16) if env.board[i // 4][i % 4] == 0]

            # Epsilon-Greedy 정책에 따라 행동 선택
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = agent(state_tensor, phase).cpu().numpy()
                    # 유효하지 않은 행동의 Q값을 -무한대로 설정
                    invalid_actions = [i for i in range(16) if i not in valid_actions]
                    q_values[invalid_actions] = -np.inf
                    action = np.argmax(q_values)

            # 현재 플레이어 저장
            current_player = env.current_player

            # 로그 기록을 위한 정보 저장
            if phase == 'select_piece':
                selected_piece = env.all_pieces[action]
                log_entry = f"P{current_player}: Selected piece {selected_piece}"
            else:
                row, col = divmod(action, 4)
                log_entry = f"P{current_player}: Placed piece at ({row}, {col})"

            # 환경에 행동 적용
            next_state, reward, done = env.step(action)

            # 로그 기록
            episode_log.append(log_entry)

            # 경험 저장
            replay_buffer.append((state, action, reward, next_state, done, phase))
            state = next_state

            # 학습
            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                batch_states = np.array([s for s, _, _, _, _, _ in minibatch])
                batch_actions = np.array([a for _, a, _, _, _, _ in minibatch])
                batch_rewards = np.array([r for _, _, r, _, _, _ in minibatch])
                batch_next_states = np.array([s_next for _, _, _, s_next, _, _ in minibatch])
                batch_dones = np.array([d for _, _, _, _, d, _ in minibatch], dtype=np.float32)
                batch_phases = [p for _, _, _, _, _, p in minibatch]

                # 텐서로 변환
                batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32).to(device)
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long).to(device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32).to(device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).to(device)

                q_values = []
                next_q_values = []
                for i in range(len(minibatch)):
                    q = agent(batch_states_tensor[i], batch_phases[i])
                    q_values.append(q)
                    next_q = agent(batch_next_states_tensor[i], batch_phases[i])
                    next_q_values.append(next_q)

                q_values = torch.stack(q_values)
                next_q_values = torch.stack(next_q_values)

                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                target_q_values = batch_rewards_tensor + gamma * max_next_q_values * (1 - batch_dones_tensor)

                predicted_q_values = q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)

                optimizer.zero_grad()
                loss = criterion(predicted_q_values, target_q_values)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            step_count += 1

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode+1}/{episodes}, Steps: {step_count/2}, Loss: {total_loss:.4f}, Epsilon: {epsilon:.4f}")

        # 에피소드 결과를 로그에 추가
        if env.winner == 1:
            episode_log.append("Result: P1 wins")
        elif env.winner == 2:
            episode_log.append("Result: P2 wins")
        elif env.winner == 0:
            episode_log.append("Result: Draw")
        else:
            episode_log.append("Result: Game not finished")

        # 에피소드 로그를 전체 로그에 추가
        all_logs.extend(episode_log)
        all_logs.append("")  # 에피소드 간 구분을 위해 빈 줄 추가

        # 100개 에피소드마다 로그를 파일로 저장
        if (episode + 1) % 100 == 0 or (episode + 1) == episodes:
            start_ep = episode - (episode % 100) + 1
            end_ep = episode + 1
            log_file_path = os.path.join(log_dir, f"episodes_{start_ep}_{end_ep}.txt")
            with open(log_file_path, 'w', encoding='utf-8') as f:
                for entry in all_logs:
                    f.write(entry + '\n')
            # 로그 리스트 초기화
            all_logs = []

    print("학습이 완료되었습니다.")
