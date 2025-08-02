import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load log probabilities
filename = 'rewards14thexperiment_contactP_entropy_logprob.pickle'
with open(filename, 'rb') as f:
    raw_log_probs = pickle.load(f)

print(f'Type of raw_log_probs: {type(raw_log_probs)}')

# Convert each item to numpy array safely
raw_log_probs_np_list = []
for r in raw_log_probs:
    if torch.is_tensor(r):
        r_cpu = r.detach().cpu().numpy()
    elif isinstance(r, (list, tuple)) and all(torch.is_tensor(x) for x in r):
        r_cpu = np.array([x.detach().cpu().numpy() for x in r])
    else:
        r_cpu = np.asarray(r)
    raw_log_probs_np_list.append(r_cpu)

# Convert to numpy array with object dtype (in case of uneven shapes)
raw_log_probs = np.array(raw_log_probs_np_list, dtype=object)

print(f'Raw log_probs shape (pre-processing): {raw_log_probs.shape}')
print(f'First entry shape: {raw_log_probs[0].shape if isinstance(raw_log_probs[0], np.ndarray) else "not array"}')

# Ensure all episodes are (steps, 7)
raw_log_probs_np = np.array(raw_log_probs)
print(f"Final shape: {raw_log_probs_np.shape}")  # Should be (nsteps, 51)

# Average over time steps
mean_log_probs_per_episode = raw_log_probs_np.mean(axis=1)  # shape (7001,)

# Plot
plt.figure(figsize=(10, 5))
episodes = np.arange(1, len(mean_log_probs_per_episode) + 1)
plt.plot(episodes, mean_log_probs_per_episode, label='Mean Log Prob per Episode')

plt.xlabel('Episode')
plt.ylabel('Mean Log Probability')
plt.title('Mean Log Probability Over Episodes')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.legend()
plt.show()