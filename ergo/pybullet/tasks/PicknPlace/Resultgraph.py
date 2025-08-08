import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

# Parameters
filename = 'rewards2.pickle'
window_size = 100

def recursive_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, list):
        return [recursive_to_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    else:
        return obj

# 1. Load the raw rewards
with open(filename, 'rb') as f:
    raw_rewards = pickle.load(f)

# 2. Convert to numpy safely
raw_rewards = recursive_to_numpy(raw_rewards)
raw_rewards = np.array(raw_rewards, dtype=np.float64)

# 3. Clean up NaNs or infinities
raw_rewards = np.nan_to_num(raw_rewards, nan=0.0, posinf=1e6, neginf=-1e6)

# 4. Cap the raw rewards for plotting
capped_rewards = np.minimum(raw_rewards, 40)
capped_rewards = np.maximum(raw_rewards,-40)

# 5. Compute the running average (over the window size), using convolution (safer)
if len(raw_rewards) >= window_size:
    running_avg = np.convolve(raw_rewards, np.ones(window_size) / window_size, mode='valid')
else:
    running_avg = np.array([])

# 6. Plot
plt.figure(figsize=(12, 6))
plt.plot(capped_rewards, label='Capped Rewards (<=100)', alpha=0.5)

if running_avg.size > 0:
    plt.plot(range(window_size - 1, len(raw_rewards)),
             running_avg,
             label=f'Running Avg (window={window_size})',
             color='red', linewidth=2)
else:
    print("Warning: Not enough data points to compute running average.")

plt.title('Reward over Time')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()