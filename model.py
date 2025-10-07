import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared_layers(state)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=1.0, c_v=0.5, c_s=0.01):
        self.gamma = gamma
        self.c_v = c_v
        self.c_s = c_s
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits, _ = self.network(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        return action.item()

    def update(self, transitions):
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get values and action distributions for the buffered states
        action_logits, values = self.network(states)
        values = values.squeeze()
        dist = Categorical(logits=action_logits)

        # Get value of the last next_state for bootstrapping
        _, next_value = self.network(next_states[-1].unsqueeze(0))
        next_value = next_value.squeeze().detach()

        # Calculate N-step returns
        returns = []
        G = next_value * (1 - dones[-1])
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns)

        # Calculate losses
        advantage = (returns - values).detach()
        
        # Actor loss
        log_probs = dist.log_prob(actions)
        actor_loss = -(advantage * log_probs).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus
        entropy_loss = dist.entropy().mean()
        
        # Combined loss
        loss = actor_loss + self.c_v * critic_loss - self.c_s * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
