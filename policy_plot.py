import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_fig(policy, algorithm, i, vmin, vmax):

    caption_fontsize = 18
    label_fontsize = 18
    tick_fontsize = 16
    cbar_tick_fontsize = 14

    policy = policy.astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(policy, cmap='viridis',
                annot=True, fmt='d', ax=ax, vmin=vmin, vmax=vmax)
    # Set the color bar font size using cbar_kws
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel('Location 2', fontsize=label_fontsize)
    ax.set_ylabel('Location 1', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.savefig('policy_' + str(i) + '_' + algorithm + '.pdf')
    plt.close()


algorithm = 'vi'
policy_file = 'results/policy_' + algorithm + '.npy'
vi_policy = np.load(policy_file)

algorithm = 'ppo'
policy_file = 'results/policy_' + algorithm + '.npy'
ppo_policy = np.load(policy_file)

for i in range(3):
    # Calculate the common colorbar range for both plots
    combined_data = np.concatenate((vi_policy[:,:,i], ppo_policy[:,:,i]))
    vmin = combined_data.min()
    vmax = combined_data.max()

    plot_fig(vi_policy[:,:,i], 'vi',i, vmin, vmax)
    plot_fig(ppo_policy[:,:,i], 'ppo',i, vmin, vmax)