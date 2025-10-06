import numpy as np
import gymnasium as gym
from model import ActorCriticAgent

def train(env, agent, episodes, N):
    episode_rewards = []
    
    # Pre-calculate scaling factors for state variables
    state_high = env.observation_space.high
    state_low = env.observation_space.low
    
    # Handle infinite values if they exist
    state_high[state_high == np.inf] = 1.0
    state_low[state_low == -np.inf] = -1.0
    
    state_range = state_high - state_low
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = (state - state_low) / state_range  # Scale the state
        
        total_reward = 0
        done = False
        transitions = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = (next_state - state_low) / state_range  # Scale the next state
            
            transitions.append((state, action, reward, next_state, done))
            
            if len(transitions) == N or done:
                agent.update(transitions)
                transitions = []
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
            
    return episode_rewards
