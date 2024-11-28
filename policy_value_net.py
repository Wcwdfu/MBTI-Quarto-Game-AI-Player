# policy_value_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicyValueNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)
        self.to(device)

    def forward(self, x):
        x = self.fc(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
