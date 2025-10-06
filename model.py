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

    def update(self, state, action, reward, next_state, done):
        active_tiles = self.get_active_tiles(state)
        
        # Calculate TD Error
        current_val = self.get_value(active_tiles)
        if done:
            next_val = 0
        else:
            next_active_tiles = self.get_active_tiles(next_state)
            next_val = self.get_value(next_active_tiles)
        
        td_error = reward + self.gamma * next_val - current_val
        
        # Critic update: Only update the weights for the active tiles
        self.w[active_tiles] += self.beta * td_error
        
        # Actor update
        prefs = self.get_policy_prefs(active_tiles)
        exp_prefs = np.exp(prefs - np.max(prefs))
        policy = exp_prefs / np.sum(exp_prefs)
        
        # Update only the columns of theta corresponding to the active tiles
        for i in range(self.action_dim):
            if i == action:
                self.theta[active_tiles, i] += self.alpha * td_error * (1 - policy[i])
            else:
                self.theta[active_tiles, i] += self.alpha * td_error * (-policy[i])
