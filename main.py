import gymnasium as gym
import cv2
import numpy as np
from model import ActorCriticAgent
from train import train
from plot import plot_training_progress
import os
import torch
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def visualize_agent(agent):
    """Renders the environment to visualize the agent's performance."""
    env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps = 2000)
    state, _ = env.reset()
    done = False
    
    while not done:
        frame = env.render()

        # Get action probabilities from the agent's network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action_logits, _ = agent.network(state_tensor)
        probs = F.softmax(action_logits, dim=-1).squeeze().cpu().detach().numpy()
        action = np.random.choice(len(probs), p=probs)
        
        # Display probabilities on the frame
        prob_text_left = f"Left: {probs[0]:.3f}"
        prob_text_right = f"Right: {probs[1]:.3f}"
        color_left = (0, 255, 0) if probs[0] > probs[1] else (0, 0, 255)  # Green if max, else Red
        color_right = (0, 255, 0) if probs[1] > probs[0] else (0, 0, 255) # Green if max, else Red

        cv2.putText(frame, prob_text_left, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)
        cv2.putText(frame, prob_text_right, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)
        
        cv2.imshow('CartPole - A2C Agent', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        
    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    env = gym.make('CartPole-v1', max_episode_steps = 2000)
    
    # Hyperparameters
    EPISODES = 2000
    N_STEPS = 16 # Collect 128 steps of experience before updating
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCriticAgent(state_dim, action_dim)
    
    scores = train(env, agent, episodes=EPISODES, N=N_STEPS)
    
    # Plot and save learning curve
    plot_training_progress(scores)
    x = [i+1 for i in range(len(scores))]
    
    # Visualize the trained agent
    visualize_agent(agent)
    
    env.close()
