import gymnasium as gym
from stable_baselines3 import PPO
import gym_examples
import numpy as np

# save policy
def plot_policy(env, model):
    
    high = env.observation_space.high
    low = env.observation_space.low

    policy = np.zeros((high[0]+1,high[1]+1, 3), dtype = int)

    for i in range(high[0]+1):
        for j in range(high[1]+1):
            policy[i,j,:], _states = model.predict(np.array([i,j]), deterministic = True)
            policy[i,j,:] = policy[i,j,:] - np.array([0,0,5])
    
    return policy

# load PPO policy
model = PPO.load("ppo")

env = gym.make('gym_examples/LateralTransEnv-v0')

policy = plot_policy(env, model)
np.save('results/policy_ppo', policy)