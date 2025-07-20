import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import ale_py

# Function to create the Atari environment with the same wrapper as in train.py
def create_atari_play_env():
    """Create Atari environment for playing with proper wrappers and human render mode."""
    try:
        # Suppress ALE logger warnings that can be noisy during play
        import ale_py
        ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    except ImportError:
        pass # ale_py might not be strictly needed if only gymnasium is used directly

    # Use the same environment ID as in train.py
    # Important: Set render_mode="human" to display the game GUI
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print("Playing in environment: ALE/Breakout-v5")
    # Apply the same AtariWrapper as in training for consistent observations
    env = AtariWrapper(env)
    return env

def play_trained_agent(model_path="dqn_model.zip", num_episodes=5):
    """
    Loads the trained DQN model and plays the Atari game.

    Args:
        model_path (str): Path to the saved DQN model.
        num_episodes (int): Number of episodes to play and visualize.
    """
    print(f"Loading trained model from: {model_path}")
    
    # 1. Set Up the Environment
    env = create_atari_play_env()

    try:
        # 2. Load the Trained Model
        # It's good practice to pass the environment when loading, especially for certain wrappers or policies.
        model = DQN.load(model_path, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'dqn_model.zip' exists and was saved correctly by train.py.")
        env.close()
        return

    print(f"\nStarting to play {num_episodes} episodes...")

    # 3. Use GreedyQPolicy and Display the Game
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"--- Episode {episode + 1}/{num_episodes} ---")
        
        # Max steps per episode to prevent infinite loops in case the agent gets stuck
        # For Breakout, typically an episode ends after losing all lives.
        max_episode_steps = 2000 # A reasonable cap, adjust if needed

        while not done and steps < max_episode_steps:
            # Predict action: deterministic=True ensures GreedyQPolicy (selects max Q-value)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # An episode ends if terminated (game over) or truncated (max steps reached)
            total_reward += reward
            steps += 1
            
            # Display the game on a GUI
            env.render()
            
            # Optional: Add a small delay to control playback speed if needed
            # time.sleep(0.01)

        print(f"Episode {episode + 1} finished.")
        print(f"Total reward for episode: {total_reward}")
        print(f"Steps taken in episode: {steps}")
        if done:
            print("Episode ended.")
        
        # Give a short break between episodes for better visualization
        time.sleep(1) 

    # 4. Close the Environment
    env.close()
    print("\nFinished playing all episodes. Environment closed.")

if __name__ == "__main__":
    play_trained_agent()