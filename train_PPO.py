import gymnasium as gym
from stable_baselines3 import PPO
import gym_examples
import numpy as np

# Parallel environments
env = gym.make('gym_examples/LateralTransEnv-v0')

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
print(model.policy)
model.learn(total_timesteps=1000000, progress_bar=True)
model.save("ppo")