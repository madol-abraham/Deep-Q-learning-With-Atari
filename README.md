# Deep Q-Learning with Stable Baselines3

This project implements a DQN agent to play Atari Breakout using Stable Baselines3 and Gymnasium.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

Run the training script:
```bash
python train.py
```

This will:
- Train a DQN agent with CNN policy on Breakout
- Save the model as `dqn_model.zip`
- Log training progress to tensorboard

### Playing with the Trained Agent

Run the play script:
```bash
python play.py
```

This will:
- Load the trained model
- Play 3 episodes with visualization
- Display performance statistics

### Hyperparameter Experiments

To run hyperparameter tuning experiments, modify `train.py` and uncomment:
```python
hyperparameter_experiments()
```

Or run individual experiments by calling the training function with different parameters.

### Policy Comparison

To compare MLP vs CNN policies, uncomment in `train.py`:
```python
compare_policies()
```

## Files

- `train.py`: Main training script with hyperparameter tuning
- `play.py`: Script to load and evaluate trained models
- `hyperparameter_table.py`: Helper to document experiment results
- `requirements.txt`: Required dependencies

## Hyperparameter Tuning

The following hyperparameters can be tuned:
- **Learning Rate (lr)**: Controls how fast the agent learns
- **Gamma (γ)**: Discount factor for future rewards
- **Batch Size**: Number of experiences sampled for each update
- **Epsilon parameters**: Control exploration vs exploitation

## Notes

- CNN Policy is recommended for Atari environments (image-based)
- Training may take significant time depending on timesteps
- Use Ctrl+C to stop training or playing early
- Tensorboard logs are saved to `./dqn_tensorboard/`

