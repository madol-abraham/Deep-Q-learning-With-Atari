import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import ale_py

# creating  Atari environment with train.py 
def create_atari_play_env():
    """Create Atari environment for playing with proper wrappers and human render mode."""
    try:
        
        import ale_py
        ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    except ImportError:
        pass 
    
    # Set render_mode="human" to display the game GUI
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print("Playing in environment: ALE/Breakout-v5")
    
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
    
    # Setting Up the Environment
    env = create_atari_play_env()

    try:
        # Load the Trained Model
        
        model = DQN.load(model_path, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'dqn_model.zip' exists and was saved correctly by train.py.")
        env.close()
        return

    print(f"\nStarting to play {num_episodes} episodes...")

    # Using GreedyQPolicy and Display the Game
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"--- Episode {episode + 1}/{num_episodes} ---")
        
        max_episode_steps = 2000 

        while not done and steps < max_episode_steps:
           
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
            total_reward += reward
            steps += 1
            
            # Display the game on a GUI
            env.render()
            
            

        print(f"Episode {episode + 1} finished.")
        print(f"Total reward for episode: {total_reward}")
        print(f"Steps taken in episode: {steps}")
        if done:
            print("Episode ended.")
        
        # Give a short break between episodes for better visualization
        time.sleep(1) 

    # Close the Environment
    env.close()
    print("\nFinished playing all episodes. Environment closed.")

if __name__ == "__main__":
    play_trained_agent()