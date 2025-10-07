import numpy as np
import gymnasium as gym
from model import ActorCriticAgent

def train(env, agent, episodes, N):
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        
        total_reward = 0
        done = False
        transitions = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
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
