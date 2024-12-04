
import torch
import torch.nn as nn
import torch.nn.functional as F

# Capturing better complex patterns in market data

#Add more layers and neurons.
#Use regularization techniques like dropout.

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.gru = nn.GRU(128, 64, batch_first=True)  # Recurrent layer
        self.attention = nn.Linear(64, 64)  # Attention mechanism
        self.fc3 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, _ = self.gru(x.unsqueeze(0))
        x = torch.tanh(self.attention(x.squeeze(0)))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
