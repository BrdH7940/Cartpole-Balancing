import gymnasium as gym
import cv2
import numpy as np
from model import ActorCriticAgent
from train import train
from plot import plot_training_progress

def visualize_agent(agent):
    print("\nVisualizing trained agent's performance...")
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=2000)
    
    # Scaling factors
    state_high = env.observation_space.high
    state_low = env.observation_space.low
    state_high[state_high == np.inf] = 1.0
    state_low[state_low == -np.inf] = -1.0
    state_range = state_high - state_low

    state, _ = env.reset()
    state = (state - state_low) / state_range
    
    total_reward = 0
    done = False
    
    while not done:
        frame = env.render()
        
        # Get action probabilities
        probs = agent.get_policy_probabilities(state)
        action = np.random.choice(len(probs), p=probs)

        # Display probabilities
        prob_text_left = f"Left: {probs[0]:.3f}"
        prob_text_right = f"Right: {probs[1]:.3f}"
        color_left = (0, 255, 0) if probs[0] > probs[1] else (0, 0, 255)
        color_right = (0, 255, 0) if probs[1] > probs[0] else (0, 0, 255)

        cv2.putText(frame, prob_text_left, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)
        cv2.putText(frame, prob_text_right, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)
        
        cv2.imshow('CartPole with Policy', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = (next_state - state_low) / state_range
        state = next_state
        total_reward += reward
        
    print(f"Total reward during visualization: {total_reward}")
    env.close()
    cv2.destroyAllWindows()

def main():
    # Environment setup
    env = gym.make('CartPole-v1', max_episode_steps = 2000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters
    EPISODES = 1000
    ALPHA = 0.03   # Actor learning rate
    BETA = 0.07    # Critic learning rate
    GAMMA = 1.0  # Discount factor
    N = 10 # N-step updates

    # Initialize agent
    agent = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA
    )

    # Train the agent
    episode_rewards = train(env, agent, EPISODES, N)

    # Plot results
    plot_training_progress(episode_rewards)

    # Visualize the trained agent
    visualize_agent(agent)

    env.close()

if __name__ == '__main__':
    main()
