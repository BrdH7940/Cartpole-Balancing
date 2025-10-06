import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(episode_rewards, window=100, solved_threshold=475):
    """
    Plots the training progress with advanced metrics to visualize stability.
    
    Args:
        episode_rewards (list or np.array): A list of total rewards for each episode.
        window (int): The size of the moving window for averaging and std dev.
        solved_threshold (int): The reward threshold for considering the environment solved.
                                 For CartPole-v1, this is typically 475.
    """
    print(f"Plotting training progress for {len(episode_rewards)} episodes...")
    
    plt.figure(figsize=(14, 8))
    
    # Plot the raw reward per episode (semi-transparent)
    plt.plot(episode_rewards, label='Total Reward per Episode', color='c', alpha=0.6, linewidth=0.8)
    
    # Calculate moving statistics
    moving_avg = []
    moving_std = []
    episodes = range(len(episode_rewards))
    
    for i in episodes:
        start = max(0, i - window + 1)
        window_rewards = episode_rewards[start:i+1]
        moving_avg.append(np.mean(window_rewards))
        moving_std.append(np.std(window_rewards))
        
    moving_avg = np.array(moving_avg)
    moving_std = np.array(moving_std)
    
    # Plot the moving average
    plt.plot(episodes, moving_avg, color='r', linestyle='-', label=f'{window}-episode Moving Average')
    
    # Plot the Standard Deviation as a shaded region, directly visualizes variance and instability
    plt.fill_between(episodes, moving_avg - moving_std, moving_avg + moving_std, 
                     color='orange', alpha=0.3, label=f'{window}-episode Std Dev')

    plt.title("Advanced Actor-Critic Training Analysis on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # Set y-axis limits for better readability, especially if there are outliers
    # The max score for CartPole-v1 is 500.
    plt.ylim(0, 550)
    
    plt.show()
