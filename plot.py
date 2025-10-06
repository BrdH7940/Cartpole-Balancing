import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(episode_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Total Reward per Episode')
    moving_avg = [np.mean(episode_rewards[max(0, i-100):i+1]) for i in range(len(episode_rewards))]
    plt.plot(moving_avg, color='red', linestyle='--', label='100-episode Moving Average')
    plt.title("Actor-Critic Training Progress on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
