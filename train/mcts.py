import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.unexplored_actions = None

    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0

    def best_child(self, exploration_weight=1.0):
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * (2 * np.log(self.visits) / (child.visits + 1e-6))**0.5
        )

class MCTS:
    def __init__(self, env, simulations=100):
        self.env = env
        self.simulations = simulations

    def search(self, initial_state):
        root = Node(initial_state)
        root.unexplored_actions = self.env.get_legal_actions(initial_state)

        for _ in range(self.simulations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return root.best_child(exploration_weight=0).state

    def _select(self, node):
        while not node.is_fully_expanded():
            if len(node.unexplored_actions) > 0:
                return self._expand(node)
            else:
                node = node.best_child()
        return node

    def _expand(self, node):
        action = node.unexplored_actions.pop()
        next_state, reward, done = self.env.simulate_action(node.state, action)
        child_node = Node(next_state, parent=node)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state):
        self.env.reset_to_state(state)
        done = False
        total_reward = 0
        while not done:
            action = self.env.sample_random_action()
            _, reward, done = self.env.step(action)
            total_reward += reward
        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
