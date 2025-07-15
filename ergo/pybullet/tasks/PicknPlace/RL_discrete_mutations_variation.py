import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
# ==== Q-Network Definition ====
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, action_dim)

    def forward(self, state, open_mask):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        masked_q = q_values.masked_fill(~open_mask.bool(), float('-inf'))
        probs = F.softmax(masked_q, dim=-1)
        return masked_q, probs

# ==== Placeholder Functions ====
def is_connected(positions, half_size=0.0075, tol=1e-4):
    from collections import deque
    positions = np.array(positions)
    visited = set()
    queue = deque([0])
    visited.add(0)
    while queue:
        current = queue.popleft()
        for i in range(len(positions)):
            if i not in visited:
                delta = np.abs(positions[current] - positions[i])
                if np.sum(np.isclose(delta, [0, 0, 2 * half_size], atol=tol)) == 1 and \
                   np.sum(delta < 2 * half_size + tol) >= 2:
                    visited.add(i)
                    queue.append(i)
    return len(visited) == len(positions)

def get_open_positions(obj_positions, voxel_size=0.015):
    existing = set(tuple(p) for p in obj_positions)
    directions = np.array([
        [voxel_size, 0, 0],
        [-voxel_size, 0, 0],
        [0, voxel_size, 0],
        [0, -voxel_size, 0],
        [0, 0, voxel_size],
        [0, 0, -voxel_size]
    ])

    open_positions = set()
    for pos in obj_positions:
        for d in directions:
            neighbor = tuple(pos + d)
            if neighbor not in existing:
                open_positions.add(neighbor)

    # Sort by X ascending
    sorted_positions = sorted(open_positions, key=lambda p: p[0])
    return [np.array(pos) for pos in sorted_positions]


# ==== Main Training Loop ====
def main():
    from MultObjPick import Obj
    from BaselineLearner import AttemptGrips

    voxel_size = 0.015
    half_size = voxel_size / 2
    n_parts = 45
    state_dim = n_parts * 3
    max_action_dim = n_parts * 6

    q_net = QNetwork(state_dim, max_action_dim)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    gamma = 0.99
    epsilon = 0.1

    dims = voxel_size * np.ones(3) / 2
    rgb = [[.75, .25, .25]] * n_parts

    used_mutations = set()
    mutation_pool = []

    max_duration_hours = 100  # Run for x hour
    start_time = time.time()
    episode = 0

    while (time.time() - start_time) < max_duration_hours * 3600:
        if not mutation_pool:
            while True:
                obj = Obj(dims, n_parts, rgb)
                obj.GenerateObject(dims, n_parts, [0, 0, 0])
                obj.isMutant = False
                obj.rgb = [tuple(color) for color in obj.rgb]  # Ensure proper format for pybullet
                parent_reward, parent_grasps = AttemptGrips(obj, 0)
                if parent_grasps > 0:
                    break
            used_mutations.clear()
            mutation_pool = get_open_positions(obj.positions,voxel_size)

        state = torch.tensor(obj.positions.flatten(), dtype=torch.float32)
        open_positions = [p for p in mutation_pool if tuple(p) not in used_mutations]
        valid_len = len(open_positions)

        if valid_len < 1:
            print("All mutations tried for current parent. Generating new parent...")
            mutation_pool.clear()
            continue

        open_positions += [open_positions[-1]] * (max_action_dim - valid_len)
        mask = torch.zeros(max_action_dim)
        mask[:valid_len] = 1

        with torch.no_grad():
            _, probs = q_net(state, mask)

        if random.random() < epsilon:
            action_idx = random.choice(torch.where(mask == 1)[0]).item()
        else:
            action_idx = torch.argmax(probs).item()

        selected_pos = open_positions[action_idx]
        used_mutations.add(tuple(selected_pos))

        mutated_obj = obj.clone_and_mutate_to(selected_pos)  # Implement this method in Obj

        if not is_connected(mutated_obj.positions):
            reward = -10.0
            num_grasps = 999
        else:
            _, num_grasps = AttemptGrips(mutated_obj, 0)
            reward = 1-num_grasps  # Lower grasps = higher reward

        next_state = torch.tensor(mutated_obj.positions.flatten(), dtype=torch.float32)
        with torch.no_grad():
            next_q, _ = q_net(next_state, mask)
            max_next_q = torch.max(next_q)

        q_values, _ = q_net(state, mask)
        target = reward + gamma * max_next_q
        loss = F.mse_loss(q_values[action_idx], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Ep {episode} | Reward: {reward:.2f} | Grasps: {num_grasps} | Mutated Pos: {selected_pos}")
        episode+=1

if __name__ == "__main__":
    main()
