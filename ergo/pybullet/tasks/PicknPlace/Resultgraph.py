import pickle
import matplotlib.pyplot as plt
import numpy as np

# Parameters
filename = 'rewards9thexperiment_contactP_entropy_rewards.pickle'  # Replace with your actual filename
window_size = 100           # Adjust this as needed

# 1. Load the raw rewards
with open(filename, 'rb') as f:
    raw_rewards = pickle.load(f)

raw_rewards = np.array(raw_rewards)

# 2. Cap the raw rewards for plotting
capped_rewards = np.minimum(raw_rewards, 50)

# 3. Compute the running average (over the window size), unclipped
cumulative_sum = np.cumsum(np.insert(raw_rewards, 0, 0))
running_avg = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

# 4. Plot
plt.figure(figsize=(12, 6))
plt.plot(capped_rewards, label='Capped Rewards (<=10)', alpha=0.5)
plt.plot(range(window_size-1, len(raw_rewards)), running_avg,
         label=f'Running Avg (window={window_size}) (unclipped)',
         color='red', linewidth=2)

plt.title('Reward over Time')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()
print("")