import numpy as np
import matplotlib.pyplot as plt

episode_rewards = np.load("episode_rewards.npy")
episode_steps = np.load("episode_steps.npy")

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def moving_std(data, window=100):
    ma = moving_average(data, window)
    # Calculate std using sliding window approach
    std = []
    half_w = window // 2
    for i in range(len(data) - window + 1):
        std.append(np.std(data[i:i+window]))
    return np.array(std)

ma_rewards = moving_average(episode_rewards, window=100)
ma_steps = moving_average(episode_steps, window=100)

std_rewards = moving_std(episode_rewards, window=100)
std_steps = moving_std(episode_steps, window=100)

episodes = np.arange(len(ma_rewards))

plt.figure(figsize=(14,6))
plt.title("Evaluation Performance (Moving Average over 100 Episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward & Steps")

plt.plot(episodes, ma_steps, label="Steps (MA 100)", color="tab:blue")
plt.fill_between(episodes, ma_steps - std_steps, ma_steps + std_steps, alpha=0.2, color="tab:blue")

plt.plot(episodes, ma_rewards, label="Rewards (MA 100)", color="tab:orange")
plt.fill_between(episodes, ma_rewards - std_rewards, ma_rewards + std_rewards, alpha=0.2, color="tab:orange")

plt.legend()
plt.grid(True)
plt.show()
