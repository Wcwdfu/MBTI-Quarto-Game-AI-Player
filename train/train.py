import torch
from .dqnagent import DQNAgent, train_dqn_with_replay
from .quartoenv import QuartoEnv

# 환경과 에이전트 초기화
state_size = 34  # 보드 상태(16) + 남은 말(16) + 선택된 말(1) + 단계(1)
action_size_select = 16  # 최대 남은 말 수
action_size_place = 16   # 보드의 위치 수

env = QuartoEnv()
agent = DQNAgent(state_size=state_size, action_size_select=action_size_select, action_size_place=action_size_place)

# 학습 실행
train_dqn_with_replay(env, agent, episodes=100)

# 학습이 완료된 후 모델 저장
torch.save(agent.state_dict(), "dqn_model.pth")
print("학습된 모델이 dqn_model.pth로 저장되었습니다.")

'''
실행은 python -m train.train

0: E  1: I 
0: N  1: S 
0: T  1: F 
0: P  1: J 
'''

