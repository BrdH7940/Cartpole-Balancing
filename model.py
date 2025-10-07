import numpy as np
from tile_coding import IHT, tiles

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, alpha=0.1, beta=0.5, gamma=0.99):
        # Tile coding setup
        self.num_tilings = 8
        self.iht_size = 4096
        self.iht = IHT(self.iht_size)
        
        # Actor and Critic parameters
        self.theta = np.zeros((self.iht_size, action_dim))  # Actor weights
        self.w = np.zeros(self.iht_size)                   # Critic weights
        
        # A common heuristic is to set learning rates relative to the number of tilings
        self.alpha = alpha / self.num_tilings # Actor learning rate
        self.beta = beta / self.num_tilings   # Critic learning rate
        self.gamma = gamma                    # Discount factor
        self.action_dim = action_dim

    def get_active_tiles(self, state):
        """ Returns the list of active tile indices for a given state. """
        # Assuming state is already scaled to [0, 1] for all dimensions
        scaled_state = [s * 10 for s in state] # Scale to [0, 10] for better tile distribution
        return tiles(self.iht, self.num_tilings, scaled_state)

    def get_policy_prefs(self, active_tiles):
        """ Calculates policy preferences (logits) by summing weights of active tiles. """
        return np.sum(self.theta[active_tiles], axis=0)

    def get_policy_probabilities(self, state):
        """ Calculates softmax policy probabilities. """
        active_tiles = self.get_active_tiles(state)
        prefs = self.get_policy_prefs(active_tiles)
        exp_prefs = np.exp(prefs - np.max(prefs)) # Softmax for stability
        return exp_prefs / np.sum(exp_prefs)

    def get_value(self, active_tiles):
        """ Calculates the state value by summing weights of active tiles. """
        return np.sum(self.w[active_tiles])

    def choose_action(self, state):
        probs = self.get_policy_probabilities(state)
        return np.random.choice(self.action_dim, p=probs)

    def update(self, transitions):
        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 1. Calculate N-step returns (iterating backwards)
        returns = []
        G = 0
        
        # Bootstrap from the value of the state just beyond the buffer
        if not dones[-1]:
            last_state_tiles = self.get_active_tiles(next_states[-1])
            G = self.get_value(last_state_tiles)
        
        # Iterate from the last transition to the first
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 2. Perform updates for each step (iterating forwards)
        for i, G in enumerate(returns):
            state = states[i]
            action = actions[i]
            
            active_tiles = self.get_active_tiles(state)
            
            # Calculate advantage
            current_val = self.get_value(active_tiles)
            advantage = G - current_val
            
            # Critic update
            self.w[active_tiles] += self.beta * advantage
            
            # Actor update
            prefs = self.get_policy_prefs(active_tiles)
            exp_prefs = np.exp(prefs - np.max(prefs))
            policy = exp_prefs / np.sum(exp_prefs)
            
            for j in range(self.action_dim):
                if j == action:
                    self.theta[active_tiles, j] += self.alpha * advantage * (1 - policy[j])
                else:
                    self.theta[active_tiles, j] += self.alpha * advantage * (-policy[j])
