import numpy as np
import gymnasium as gym
import gym_examples
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO

env = gym.make('gym_examples/LateralTransEnv-v0')

# load DP policy
policy = np.load('results/policy_vi.npy')

# load PPO policy
model = PPO.load("ppo")

# state probability
state_prob_dp = np.zeros((env.max_onhand_qty_1+1, env.max_onhand_qty_2+1))
state_prob_ppo = np.zeros((env.max_onhand_qty_1+1, env.max_onhand_qty_2+1))

# test setting
num_episodes = 50000

# test DP policy
obs, _ = env.reset()
total_reward = 0

for i in range(num_episodes):
    action = policy[obs[0], obs[1], :]
    action_mask = action + [0,0,5]
    obs, rewards, truncation, dones, info = env.step(action_mask)
    total_reward += rewards
    state_prob_dp[obs[0], obs[1]] += 1

average_reward = total_reward / num_episodes
print("Average reward: ", average_reward)

# test PPO policy
obs, _ = env.reset()
total_reward = 0

for i in range(num_episodes):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, truncation, dones, info = env.step(action)
    total_reward += rewards
    state_prob_ppo[obs[0], obs[1]] += 1

average_reward = total_reward / num_episodes
print("Average reward: ", average_reward)

# normalize state probability
state_prob_dp /= num_episodes
state_prob_ppo /= num_episodes

# Calculate the common colorbar range for both plots
combined_data = np.concatenate((state_prob_dp, state_prob_ppo))
vmin = combined_data.min()
vmax = combined_data.max()

caption_fontsize = 18
label_fontsize = 18
tick_fontsize = 16
cbar_tick_fontsize = 14

# heat map of state probability for DP policy
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(state_prob_dp, cmap='viridis', ax=ax, vmin=vmin, vmax=vmax)
# Set the color bar font size using cbar_kws
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
ax.invert_yaxis()
ax.set_xlabel('Location 2', fontsize=label_fontsize)
ax.set_ylabel('Location 1', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# ax.set_title('Value iteration', fontsize=20)
plt.savefig('state_visitation_dp.pdf')
plt.close()

# heat map of state probability for PPO policy
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(state_prob_ppo, cmap='viridis', ax=ax, vmin=vmin, vmax=vmax)
# Set the color bar font size using cbar_kws
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
ax.invert_yaxis()
ax.set_xlabel('Location 2', fontsize=label_fontsize)
ax.set_ylabel('Location 1', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# ax.set_title('PPO', fontsize=20)
plt.savefig('state_visitation_ppo.pdf')
plt.close()