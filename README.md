# Deep Q-Learning for Atari Breakout

This project implements a Deep Q-Network (DQN) agent to play Atari Breakout using Stable Baselines3 and Gymnasium.

## Project Overview

Deep Q-Learning combines Q-learning with deep neural networks to handle complex state spaces like Atari games. This implementation:

- Uses Stable Baselines3 DQN implementation
- Compares MLP vs CNN policies for image-based environments
- Experiments with various hyperparameters to optimize performance
- Provides visualization tools to evaluate agent performance

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Atari ROMs installed:
```bash
python -m autorom.cli --accept-license
```

## Usage

### Training the Agent

Run the training script:
```bash
python train.py
```

This will:
- Train DQN agents with different policies and hyperparameters
- Save the best performing model as `dqn_model.zip`
- Log training progress to tensorboard
- Save hyperparameter results to `training_results.csv`

### Playing with the Trained Agent

Run the play script:
```bash
python play.py
```

This will:
- Load the trained model
- Play episodes with visualization
- Display performance statistics

## Demo

https://github.com/user-attachments/assets/72b0effa-3a1e-41f9-8400-4ac3adfad33a

## Files

- `train.py`: Main training script with hyperparameter tuning
- `play.py`: Script to load and evaluate trained models
- `requirements.txt`: Required dependencies
- `training_results.csv`: Hyperparameter experiment results

## Hyperparameter Tuning Results

Our experiments tested the following hyperparameters:

| Hyperparameter Set                                                                 | Noted Behavior |
|------------------------------------------------------------------------------------|----------------|
| `lr=0.0001`, `gamma=0.99`, `batch=32`, `epsilon_start=1.0`, `epsilon_end=0.01`, `epsilon_decay=0.1` | Achieved an average reward of **1.0** which was a better performance compared to the CnnPolicy configs. However, the performance was poor suggesting minimal learning. The agent averaged 0.67 reward over 3 episodes. |
| `lr=0.0001`, `gamma=0.99`, `batch=32`, `epsilon_start=1.0`, `epsilon_end=0.02`, `epsilon_decay=0.995` | Achieved an average reward of 1.67 over 5 episodes which is an improvement compared to the previous training. The agent mostly moved the paddle randomly and missed the ball quickly. The timesteps were increased from 25000 to 100000 for better training which took longer. |
| `lr=0.0001`, `gamma=0.99`, `batch=32`, `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay=0.9` | Achieved an average reward of 2.0 with training time of 1882.29s. The agent mostly moved the paddle randomly and missed the ball quickly, although episode one had a reward of 4.0 with occasional breaking of bricks. |
| `lr=0.00025`, `gamma=0.99`, `batch=32`, `epsilon_start=1.0`, `epsilon_end=0.01`, `epsilon_decay=0.1` | This was the best performing config which achieved a total reward of 7.0 over 5 episodes. The agent showed improvement in controlling the paddle, with better performance during the second episode. Training time was approximately 556.39s, however there was still room for improvement to reach the best model |


## Findings

1. **Policy Comparison**: CNN policies significantly outperform MLP policies for image-based Atari environments due to their ability to extract spatial features.

2. **Learning Rate**: Higher learning rates (2.5e-4) can accelerate training but may lead to instability, while lower rates (1e-4) provide more stable learning.

3. **Gamma (Discount Factor)**: Higher values (0.995) encourage long-term planning, while lower values (0.99) focus more on immediate rewards.

4. **Batch Size**: Larger batch sizes (64) provide more stable gradient updates but require more computation per step compared to smaller batches (32).

5. **Exploration Strategy**: The exploration schedule significantly impacts learning, with longer exploration phases helping to discover better policies.

## Technical Implementation

The DQN implementation includes:
- Experience replay buffer to break correlations between consecutive samples
- Target network for stable Q-value updates
- Epsilon-greedy exploration strategy with annealing
- Frame stacking for temporal information
- AtariWrapper for preprocessing (frame resizing, grayscale conversion)

## Notes

- CNN Policy is recommended for Atari environments due to their image-based nature
- Training requires significant computational resources
- Tensorboard logs are saved to `./dqn_tensorboard/`
- Use Ctrl+C to stop training or playing early

## Group Contribution

1. Madol Abraham Kuol Madol -  Implemented a comprehensive DQN training framework that systematically compares neural network architectures.
2. Bernice Uwituze - Hyperparameter tuning on train.py to get best performing deep q-learning model
3. Aubin Ntwali
4. Valentine Tobechi Kalu
