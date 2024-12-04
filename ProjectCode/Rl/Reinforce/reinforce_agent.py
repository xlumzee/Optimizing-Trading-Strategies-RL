
import torch
import torch.optim as optim
from policy_network import PolicyNetwork
import random

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        # Define exploration probability (decays over time)
        exploration_prob = max(0.05, 1.0 - len(self.rewards) * 0.002)  # Slower decay


        # Epsilon-greedy exploration strategy
        if random.random() < exploration_prob:
            # Random action for exploration
            return random.choice(range(self.policy_network.fc3.out_features))
        else:
            # Policy-driven action for exploitation
            state = torch.from_numpy(state).float()
            action_probs = self.policy_network(state)
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
            self.log_probs.append(distribution.log_prob(action))
            return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        # Calculate the discounted future rewards
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)

        # Convert returns to a tensor and normalize
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy loss
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Combine log_probs into a single tensor
        log_probs_tensor = torch.stack(self.log_probs)
        entropy = -torch.sum(torch.exp(log_probs_tensor) * log_probs_tensor)  # Correct entropy calculation

        # Add entropy to the policy loss
        policy_loss = torch.stack(policy_loss).sum() - 0.01 * entropy  # Weight entropy loss

        # Backpropagate the loss
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()

        # Reset rewards and log probabilities
        self.log_probs = []
        self.rewards = []
