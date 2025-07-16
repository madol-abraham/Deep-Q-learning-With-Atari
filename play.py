import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

def play_trained_agent(model_path="dqn_model.zip", num_episodes=5, render_mode="human"):
    """Load and play with trained DQN agent"""
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = DQN.load(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first using train.py")
        return
    
    # Create the same environment used for training
    def make_env():
        env = gym.make('ALE/Breakout-v5')
        env = AtariWrapper(env)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    
    # For visualization, we need a single environment
    if render_mode == "human":
        single_env = gym.make('ALE/Breakout-v5', render_mode="human")
        single_env = AtariWrapper(single_env)
        single_env = gym.wrappers.FrameStackObservation(single_env, 4)
    
    print(f"\nPlaying {num_episodes} episodes with trained agent...")
    print("Press Ctrl+C to stop early\n")
    
    total_rewards = []
    
    try:
        for episode in range(num_episodes):
            if render_mode == "human":
                obs, _ = single_env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                while not done:
                    # Use the trained model to predict action (greedy policy)
                    # Convert observation to the format expected by the model
                    obs_array = obs.__array__() if hasattr(obs, '__array__') else obs
                    # Remove the extra dimension if present
                    if obs_array.shape == (4, 84, 84, 1):
                        obs_array = obs_array.squeeze(-1)
                    action, _ = model.predict(obs_array, deterministic=True)
                    obs, reward, terminated, truncated, _ = single_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    step_count += 1
                    
                    # Add small delay for better visualization
                    time.sleep(0.02)
                    
                    # Break if episode is too long
                    if step_count > 10000:
                        break
                
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
                
            else:
                # Use vectorized environment for faster evaluation without rendering
                obs = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward[0]
                
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        if render_mode == "human":
            single_env.close()
        else:
            env.close()
    
    # Print statistics
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        max_reward = max(total_rewards)
        min_reward = min(total_rewards)
        
        print(f"\n=== DETAILED PERFORMANCE ANALYSIS ===")
        print(f"Episodes played: {len(total_rewards)}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Min reward: {min_reward:.2f}")
        print(f"Reward standard deviation: {(sum([(r-avg_reward)**2 for r in total_rewards])/len(total_rewards))**0.5:.2f}")
        print(f"Performance consistency: {'High' if max_reward - min_reward < avg_reward else 'Variable'}")
        print(f"Agent learning evidence: {'Improving' if total_rewards[-1] > avg_reward else 'Stable'}")
        print(f"Individual episode rewards: {total_rewards}")
        
        # Performance interpretation
        print(f"\n=== AGENT PERFORMANCE INTERPRETATION ===")
        if avg_reward > 10:
            print("EXCELLENT: Agent demonstrates strong game understanding and strategy")
        elif avg_reward > 5:
            print("GOOD: Agent shows competent gameplay with room for improvement")
        elif avg_reward > 1:
            print("DEVELOPING: Agent has basic functionality but needs more training")
        else:
            print("BASIC: Agent requires significant additional training")

def evaluate_model(model_path="dqn_model.zip", num_episodes=10):
    """Comprehensive model evaluation without rendering"""
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    print("Testing agent performance across multiple episodes...")
    print("This evaluation uses GREEDY POLICY (deterministic=True) for optimal performance")
    play_trained_agent(model_path, num_episodes, render_mode="none")

if __name__ == "__main__":
    # Play with visualization
    play_trained_agent("dqn_model.zip", num_episodes=3, render_mode="human")
    
    # Optional: Run evaluation without rendering for more episodes
    # evaluate_model("dqn_model.zip", num_episodes=10)