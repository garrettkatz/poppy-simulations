import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load rewards
filename = 'rewards10thdexperiment_contactP_entropy_stddev.pickle'
with open(filename, 'rb') as f:
    raw_rewards = pickle.load(f)
print(f'Type of raw_rewards: {type(raw_rewards)}')

# Convert to numpy if necessary
raw_rewards_np_list = [r.detach().cpu().numpy() if torch.is_tensor(r) else np.asarray(r) for r in raw_rewards]
raw_rewards = np.array(raw_rewards_np_list)
print(f'Raw rewards shape: {raw_rewards.shape}')
assert raw_rewards.ndim == 2 and raw_rewards.shape[1] == 7, "Expected shape (N, 7)"

num_episodes, num_components = raw_rewards.shape

# Plot std devs over episodes directly
plt.figure(figsize=(10, 6))
episodes = np.arange(1, num_episodes + 1)
for comp in range(num_components):
    plt.plot(episodes, raw_rewards[:, comp], label=f'Component {comp + 1}')

plt.xlabel('Episode')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation per Reward Component Over Episodes')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print("")