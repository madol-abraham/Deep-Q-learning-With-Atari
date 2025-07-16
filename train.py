import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def create_atari_env():
    """Create Atari environment with proper wrappers"""
    try:
        import ale_py
        ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    except:
        pass
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    print("Using environment: ALE/Breakout-v5")
    env = AtariWrapper(env)
    return env

def train_dqn(policy_type, lr, gamma, batch_size, epsilon_start, epsilon_end, epsilon_decay, total_timesteps=25000):
    """Train DQN with given hyperparameters (Atari version)"""
    print(f"\nTraining {policy_type} policy...")
    print(f"lr={lr}, gamma={gamma}, batch_size={batch_size}")
    print(f"epsilon_start={epsilon_start}, epsilon_end={epsilon_end}, epsilon_decay={epsilon_decay}")
    env = DummyVecEnv([create_atari_env])
    model = DQN(
        policy_type,
        env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        exploration_fraction=epsilon_decay,
        exploration_initial_eps=epsilon_start,
        exploration_final_eps=epsilon_end,
        buffer_size=20000,
        learning_starts=1000,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/"
    )
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    # Evaluate the model
    env_eval = create_atari_env()
    total_reward = 0
    episodes = 3
    for episode in range(episodes):
        obs, _ = env_eval.reset()
        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_eval.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
    avg_reward = total_reward / episodes
    env_eval.close()
    return model, avg_reward, training_time

def main():
    print("DQN Training on Atari Breakout")
    print("Comparing MLPPolicy vs CNNPolicy with hyperparameter tuning")
    print("="*60)
    # Optimized hyperparameter configurations for best performance
    configs = [
        {"policy": "MlpPolicy", "lr": 1e-4, "gamma": 0.99, "batch_size": 32, "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.1},
        {"policy": "CnnPolicy", "lr": 1e-4, "gamma": 0.99, "batch_size": 32, "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.1},
        {"policy": "CnnPolicy", "lr": 2.5e-4, "gamma": 0.99, "batch_size": 32, "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.1},
        {"policy": "CnnPolicy", "lr": 1e-4, "gamma": 0.995, "batch_size": 64, "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.2}
    ]
    results = []
    best_model = None
    best_reward = -float('inf')
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Configuration {i}")
        print("-" * 40)
        model, avg_reward, training_time = train_dqn(
            policy_type=config["policy"],
            lr=config["lr"],
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config["epsilon_decay"],
            total_timesteps=25000
        )
        results.append({
            "Config": i,
            "Policy": config["policy"],
            "Learning Rate": config["lr"],
            "Gamma": config["gamma"],
            "Batch Size": config["batch_size"],
            "Epsilon Start": config["epsilon_start"],
            "Epsilon End": config["epsilon_end"],
            "Epsilon Decay": config["epsilon_decay"],
            "Avg Reward": round(avg_reward, 2),
            "Training Time (s)": round(training_time, 2)
        })
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = model
        print(f"Result: Avg Reward = {avg_reward:.2f}, Time = {training_time:.1f}s")
    # Save the best model
    if best_model:
        best_model.save("dqn_model")
        print(f"\n[SUCCESS] Best model saved as 'dqn_model.zip' (Reward: {best_reward:.2f})")
    # Display results table
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"{'='*80}")
    # Show key columns
    display_df = df[['Config', 'Policy', 'Learning Rate', 'Gamma', 'Batch Size', 'Epsilon Start', 'Epsilon End', 'Epsilon Decay', 'Avg Reward']]
    print(display_df.to_string(index=False))
    # Save detailed results
    df.to_csv("training_results.csv", index=False)
    print(f"\n[SUCCESS] Detailed results saved to 'training_results.csv'")
    print(f"[SUCCESS] Run 'python play.py' to see the trained agent in action!")



if __name__ == "__main__":
    main()