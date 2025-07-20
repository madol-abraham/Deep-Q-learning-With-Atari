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

| Hyperparameter Set | Observed Behavior |
|-------------------|-------------------|
| MlpPolicy, lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | MLP policy struggles with image-based input, resulting in poor performance |
| CnnPolicy, lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | CNN policy provides better feature extraction for images, establishing a baseline performance |
| CnnPolicy, lr=2.5e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Higher learning rate accelerates training but may lead to less stable convergence |
| CnnPolicy, lr=1e-4, gamma=0.995, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.2 | Higher gamma values future rewards more, larger batch size provides more stable updates |

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
