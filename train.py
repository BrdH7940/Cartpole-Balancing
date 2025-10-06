import numpy as np
import gymnasium as gym
from model import ActorCritic

def train(agent, episodes=1000):
    env = gym.make('CartPole-v1')
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            features = agent.get_features(state)
            action = agent.select_action(features)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward (last 50): {avg_reward:.2f}")
            
            if avg_reward >= 475.0:
                print(f"Solved at episode {episode}!")
                break
                
    env.close()
    return episode_rewards
