import pickle
import matplotlib.pyplot as plt
import os

# Path to your pickle file
pickle_file = 'A2C_rewards_sum_results.pickle'
import pickle
import matplotlib.pyplot as plt
import numpy as np
WorkingDir = os.getcwd() #+ "/Feb12Res_cond_iou_co13"
os.chdir(WorkingDir)
# Load the pickle file
with open(pickle_file, 'rb') as f:
    rewards = pickle.load(f)

# Check structure
print("Type of rewards:", type(rewards))
print("Length:", len(rewards))
print("First few entries:", rewards[:5])

# Convert to NumPy array if needed
rewards = np.array(rewards)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reinforcement Learning Rewards')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()