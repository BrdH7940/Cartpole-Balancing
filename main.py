import gymnasium as gym
import cv2
import numpy as np
from model import ActorCritic
from train import train
from plot import plot_training_progress

def visualize_agent(agent):
    print("\nVisualizing trained agent's performance for one episode...")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Render the environment frame
        frame = env.render()
        
        # Get action probabilities
        features = agent.get_features(state)
        probs = agent.policy(features)
        action = np.random.choice(agent.n_actions, p=probs) # We still sample to see the actual run

        # Prepare text to display
        prob_text_left = f"Left: {probs[0]:.3f}"
        prob_text_right = f"Right: {probs[1]:.3f}"

        # Use OpenCV to add text to the frame
        # If left probability is higher, color it red and the other green. Vice-versa.
        if probs[0] > probs[1]:
            color_left = (0, 0, 255)  # Red
            color_right = (0, 255, 0) # Green
        else:
            color_left = (0, 255, 0)  # Green
            color_right = (0, 0, 255) # Red

        cv2.putText(frame, prob_text_left, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)
        cv2.putText(frame, prob_text_right, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)
        
        # Display the frame
        cv2.imshow('CartPole with Policy', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to quit
            break

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        
    print(f"Total reward during visualization: {total_reward}")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    agent = ActorCritic(state_dim=4, n_actions=2, alpha_actor=0.001, alpha_critic=0.002)
    episode_rewards = train(agent, episodes = 2000)
    plot_training_progress(episode_rewards)
    visualize_agent(agent)
