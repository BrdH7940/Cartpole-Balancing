import numpy as np

class ActorCritic:
    def __init__(self, state_dim, n_actions, gamma=0.99, alpha_actor=0.001, alpha_critic=0.01):
        self.d = 1 + state_dim + (state_dim * (state_dim + 1)) // 2  # Feature dimension
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        
        # Parameters
        self.theta = np.random.randn(self.d, n_actions) * 0.01  # Actor
        self.w = np.random.randn(self.d) * 0.01  # Critic
    
    def get_features(self, state):
        """Convert state to polynomial features"""
        features = []
        for i in range(len(state)):
            features.append(state[i])
            for j in range(i, len(state)):
                features.append(state[i] * state[j])
        return np.array([1] + features)
    
    def policy(self, features):
        """Get action probabilities"""
        preferences = features @ self.theta
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)
    
    def select_action(self, features):
        """Sample action from policy"""
        probs = self.policy(features)
        return np.random.choice(self.n_actions, p=probs)
    
    def value(self, features):
        """State-value estimate"""
        return features @ self.w
    
    def update(self, state, action, reward, next_state, done):
        """One-step Actor-Critic update"""
        phi = self.get_features(state)
        phi_next = self.get_features(next_state) if not done else np.zeros_like(phi)
        
        # TD Error
        V_current = self.value(phi)
        V_next = self.value(phi_next) if not done else 0
        td_error = reward + self.gamma * V_next - V_current
        
        # Critic update
        self.w += self.alpha_critic * td_error * phi
        
        # Actor update
        probs = self.policy(phi)
        action_grad = np.zeros(self.n_actions)
        action_grad[action] = 1
        policy_grad = phi[:, np.newaxis] @ (action_grad - probs)[np.newaxis, :]
        self.theta += self.alpha_actor * td_error * policy_grad
