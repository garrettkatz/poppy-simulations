value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)

# --- Adaptive, Targeted Entropy Loss ---
entropy = dist.entropy()  # This gives entropy for each action dimension (batch_size, action_dim)

# Create masks for approach and pickup action indices using the global constants
approach_mask = torch.zeros_like(entropy, dtype=torch.bool)
pickup_mask = torch.zeros_like(entropy, dtype=torch.bool)

for i in range(states.shape[0]):  # Iterate through the batch
    for idx in APPROACH_ACTION_INDICES:
        approach_mask[i, idx] = True
    for idx in PICKUP_ACTION_INDICES:
        pickup_mask[i, idx] = True

# Calculate average approach and pickup rewards over the trajectory
# Use .item() to convert single-element tensors to Python scalars for operations like np.mean
avg_approach_reward = np.mean(approach_rewards_list)  # Use np.mean for lists
avg_pickup_reward = np.mean(pickup_rewards_list)

# --- Define Adaptive Beta Functions (Hyperparameters for tuning) ---

# Beta for Approach-related Joints (increases exploration when approach reward is low)
beta_approach_base = 0.005  # Base coefficient for inverse reward relationship
max_beta_approach = 0.05  # Maximum entropy coefficient
min_beta_approach = 0.001  # Minimum entropy coefficient

if avg_approach_reward > 0:
    beta_approach = beta_approach_base / (avg_approach_reward + 1e-6)
else:
    beta_approach = max_beta_approach  # Maximize exploration if reward is non-positive
beta_approach = torch.clamp(torch.tensor(beta_approach), min_beta_approach, max_beta_approach)

# Beta for Pickup-related Joints (conditional exploration)
beta_pickup_base = 0.005
max_beta_pickup = 0.05
min_beta_pickup = 0.001

# --- New: Define a threshold for what constitutes a "met" approach reward ---
# This value needs careful tuning based on your RAW approach reward scale.
# E.g., if MAX_APPROACH_REWARD (raw) in your rewards function is 100,
# then 50.0 means half of the max reward is achieved.
APPROACH_REWARD_MET_THRESHOLD = 50.0  # TUNE THIS CAREFULLY!

# Influence from Approach Reward: Only encourage pickup exploration significantly if approach reward is met
beta_pickup_from_approach = torch.tensor(
    min_beta_pickup)  # Default: no extra influence from approach unless threshold is met

if avg_approach_reward >= APPROACH_REWARD_MET_THRESHOLD:
    # If approach reward threshold IS met, then scale the influence based on how good it is.
    # Use the same normalization factor (e.g., 100.0) that corresponds to your MAX_APPROACH_REWARD (raw)
    # in the rewards function.
    normalized_approach_reward_for_beta = torch.clamp(torch.tensor(avg_approach_reward / 100.0), 0.0, 1.0)
    beta_pickup_from_approach = max_beta_pickup * normalized_approach_reward_for_beta
    beta_pickup_from_approach = torch.max(beta_pickup_from_approach, torch.tensor(min_beta_pickup))

# Influence from Pickup Reward itself: always encourage exploration if pickup reward is low
beta_pickup_from_pickup = torch.tensor(min_beta_pickup)  # Default: no extra influence
if avg_pickup_reward > 0:
    beta_pickup_from_pickup = beta_pickup_base / (avg_pickup_reward + 1e-6)
else:
    beta_pickup_from_pickup = max_beta_pickup
beta_pickup_from_pickup = torch.clamp(beta_pickup_from_pickup, min_beta_pickup, max_beta_pickup)

# Combine influences: Take the maximum of these two terms.
beta_pickup = torch.max(beta_pickup_from_approach, beta_pickup_from_pickup)
beta_pickup = torch.clamp(beta_pickup, min_beta_pickup, max_beta_pickup)

# Calculate the total conditional entropy bonus
conditional_entropy_bonus = 0.0

# Apply beta_approach to approach-related joints' entropy
if approach_mask.any():
    conditional_entropy_bonus += (entropy[approach_mask] * beta_approach).mean()

# Apply beta_pickup to pickup-related joints' entropy
if pickup_mask.any():
    conditional_entropy_bonus += (entropy[pickup_mask] * beta_pickup).mean()

# Total Loss: Policy Loss + Value Loss - Conditional Entropy Bonus
loss = policy_loss + 0.5 * value_loss - conditional_entropy_bonus

self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()