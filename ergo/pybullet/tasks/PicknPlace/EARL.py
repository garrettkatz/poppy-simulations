import argparse
import random
import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import os
from MultObjPick import Obj as SimObj
from BaselineLearner import AttemptGrips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Utilities
# -----------------------------
CACHE_FILE = "evaluated_objects.csv"
EVAL_CACHE = {}
def obj_to_key(obj):
    coords = [tuple(np.round(p,6)) for p in obj.positions]
    coords_sorted = tuple(sorted(coords))
    return str(coords_sorted)

def load_cache(filename=CACHE_FILE):
    cache = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                cache[row[0]] = float(row[1])
    return cache

def update_cache(filename, obj, fitness):
    import csv, os
    key = obj_to_key(obj)

    if key not in EVAL_CACHE:
        EVAL_CACHE[key] = fitness
        file_exists = os.path.exists(filename)

        # Use append mode, but add header only if file doesn't exist
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["object_key", "fitness"])
            writer.writerow([key, fitness])

    #return cache.writerow([key, fitness])
   # return cache
def get_open_positions(obj_positions, voxel_size=0.015, max_positions=128):
    existing = set(tuple(np.round(p,6)) for p in obj_positions)
    directions = [
        np.array([voxel_size, 0.0, 0.0]),
        np.array([-voxel_size, 0.0, 0.0]),
        np.array([0.0, voxel_size, 0.0]),
        np.array([0.0, -voxel_size, 0.0]),
        np.array([0.0, 0.0, voxel_size]),
        np.array([0.0, 0.0, -voxel_size])
    ]
    open_positions = set()
    for pos in obj_positions:
        for d in directions:
            candidate = tuple(np.round((pos + d), 6))
            if candidate not in existing:
                open_positions.add(candidate)
    sorted_open = sorted(open_positions, key=lambda p: (p[0], p[1], p[2]))
    valid_count = min(len(sorted_open), max_positions)
    padded = []
    for i in range(max_positions):
        if i < valid_count:
            padded.append(np.array(sorted_open[i], dtype=np.float32))
        else:
            padded.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return padded, valid_count

def is_connected(positions, voxel_size=0.015):
    if len(positions) <= 1:
        return True
    coords = [tuple(np.round(p,6)) for p in positions]
    neighbors = set(coords)
    offs = set([
        (voxel_size, 0.0, 0.0),
        (-voxel_size, 0.0, 0.0),
        (0.0, voxel_size, 0.0),
        (0.0, -voxel_size, 0.0),
        (0.0, 0.0, voxel_size),
        (0.0, 0.0, -voxel_size),
    ])
    visited = set()
    stack = [coords[0]]
    visited.add(coords[0])
    while stack:
        cur = stack.pop()
        for off in offs:
            nb = (round(cur[0] + off[0],6), round(cur[1] + off[1],6), round(cur[2] + off[2],6))
            if nb in neighbors and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(coords)

# -----------------------------
# Policy and Value networks
# -----------------------------
class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, max_actions=128, hidden_dim=256):
        super().__init__()
        self.max_actions = max_actions
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(hidden_dim, max_actions)

    def forward(self, state):
        h = self.shared(state)
        return self.logits_head(h)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

# -----------------------------
# Replay buffer
# -----------------------------
class DiscreteReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action_idx, reward, log_prob, entropy, value):
        self.buffer.append((state.detach().cpu(), torch.tensor(action_idx).long(), float(reward),
                            log_prob.detach().cpu(), entropy.detach().cpu(), value.detach().cpu()))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, logps, ents, vals = zip(*batch)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        logps = torch.stack(logps).to(device)
        ents = torch.stack(ents).to(device)
        vals = torch.stack(vals).to(device)
        return states, actions, rewards, logps, ents, vals
    def __len__(self):
        return len(self.buffer)

# -----------------------------
# PPO update
# -----------------------------
def ppo_update_categorical(policy, value_net, opt_policy, opt_value, buffer,
                           batch_size=16, epochs=4, clip_epsilon=0.2, c1=0.5, c2=0.01):
    if len(buffer) < batch_size:
        return 0.0, 0.0

    states, actions, rewards, old_logps, _, _ = buffer.sample(batch_size)
    returns = rewards

    with torch.no_grad():
        values = value_net(states)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss_total, value_loss_total = 0.0, 0.0

    for _ in range(epochs):
        # Policy forward pass
        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value forward pass (separate from policy)
        value_pred = value_net(states)
        value_loss = F.mse_loss(value_pred, returns)

        # Policy optimization
        opt_policy.zero_grad()
        (policy_loss - c2 * entropy).backward()  # Only policy loss + entropy
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt_policy.step()

        # Value optimization
        opt_value.zero_grad()
        (c1 * value_loss).backward()  # Only value loss
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        opt_value.step()

        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()

    return policy_loss_total / epochs, value_loss_total / epochs
def ppo_update_categorical_Old(policy, value_net, opt_policy, opt_value, buffer,
                           batch_size=32, epochs=4, clip_epsilon=0.2, c1=0.5, c2=0.01):
    if len(buffer) < batch_size:
        return 0.0, 0.0
    states, actions, rewards, old_logps, _, _ = buffer.sample(batch_size)
    returns = rewards
    with torch.no_grad():
        values = value_net(states)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss_total, value_loss_total = 0.0, 0.0
    for _ in range(epochs):
        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value_net(states), returns)
        loss = policy_loss + c1 * value_loss - c2 * entropy
        opt_policy.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt_policy.step()
        opt_value.zero_grad(); (value_loss * c1).backward(); torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5); opt_value.step()
        policy_loss_total += policy_loss.item(); value_loss_total += value_loss.item()
    return policy_loss_total / epochs, value_loss_total / epochs

# -----------------------------
# Encoding and rewards
# -----------------------------
def encode_state_for_policy(parent1, parent2, pop_objs, fitnesses, k=10):
    def encode_obj(obj, k):
        pos = np.array(obj.positions)
        if pos.size == 0:
            return [0.0] * (k * 3)
        centroid = np.mean(pos, axis=0)
        dists = np.linalg.norm(pos - centroid, axis=1)
        idx = np.argsort(dists)[:k]
        chosen = pos[idx]
        flat = chosen.flatten().tolist()
        while len(flat) < k * 3:
            flat += [0.0]
        return flat[:k * 3]
    p1_enc, p2_enc = encode_obj(parent1, k), encode_obj(parent2, k)
    p1_fit, p2_fit = parent1.fitness, parent2.fitness
    stats = [float(np.min(fitnesses)), float(np.max(fitnesses)), float(np.mean(fitnesses)), float(np.std(fitnesses))]
    state_list = p1_enc + p2_enc + [float(p1_fit), float(p2_fit)] + stats
    return torch.tensor(state_list, dtype=torch.float32, device=device)

def compute_voxel_reward(child_obj, parent1_obj, parent2_obj, pop_objs, fitnesses):
    child_fit = child_obj.fitness
    best_parent = float(min(parent1_obj.fitness, parent2_obj.fitness)) + 1e-8
    improvement = (best_parent - child_fit) / best_parent
    better_than = sum(1 for f in fitnesses if child_fit < f)
    ranking = (better_than / max(1, len(fitnesses))) - 0.5
    def centroid(obj): return np.mean(np.array(obj.positions), axis=0)
    try:
        dists = [np.linalg.norm(centroid(child_obj) - centroid(p)) for p in pop_objs]
        min_dist = float(np.min(dists)) if len(dists) > 0 else 0.0
    except Exception:
        min_dist = 0.0
    diversity = min(min_dist / 0.1, 1.0)
    return float(2.0 * improvement + 1.0 * ranking + 0.5 * diversity)

# -----------------------------
# GA operators
# -----------------------------
def voxel_crossover_mutation(parent1, parent2, mutation_rate=0.2, voxel_size=0.015):
    """
    Crossover that takes half of each parent while maintaining connectivity
    """
    from collections import deque

    child = copy.deepcopy(parent1)

    # Take half of parent1 positions
    p1_positions = list(parent1.positions)
    random.shuffle(p1_positions)
    half_p1 = p1_positions[:len(p1_positions) // 2] if len(p1_positions) > 1 else p1_positions

    # Start child with half of parent1
    child.positions = half_p1.copy()
    child_positions = set(tuple(round(coord, 6) for coord in pos) for pos in child.positions)

    directions = [
        (voxel_size, 0, 0), (-voxel_size, 0, 0),
        (0, voxel_size, 0), (0, -voxel_size, 0),
        (0, 0, voxel_size), (0, 0, -voxel_size)
    ]

    # Find voxels from parent2 that are adjacent to current child structure
    p2_positions = set(tuple(round(coord, 6) for coord in pos) for pos in parent2.positions)
    adjacent_from_p2 = set()

    for pos in child_positions:
        for dx, dy, dz in directions:
            neighbor = tuple(round(coord, 6) for coord in [pos[0] + dx, pos[1] + dy, pos[2] + dz])
            if neighbor in p2_positions and neighbor not in child_positions:
                adjacent_from_p2.add(neighbor)

    # Add voxels from parent2 that connect to the structure
    # Target: roughly half the size of parent2
    target_from_p2 = len(parent2.positions) // 2
    added_count = 0

    # Use BFS-like approach to grow connected structure from parent2
    candidates = list(adjacent_from_p2)
    random.shuffle(candidates)

    while candidates and added_count < target_from_p2:
        # Take next candidate
        pos = candidates.pop(0)
        if pos not in child_positions:
            # Add this voxel
            child.positions.append(np.array(pos))
            child_positions.add(pos)
            added_count += 1

            # Find new adjacent candidates from parent2
            for dx, dy, dz in directions:
                neighbor = tuple(round(coord, 6) for coord in [pos[0] + dx, pos[1] + dy, pos[2] + dz])
                if (neighbor in p2_positions and
                        neighbor not in child_positions and
                        neighbor not in candidates):
                    candidates.append(neighbor)

    # Mutation with connectivity preservation
    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(child.positions) > 1:
            # Remove a voxel only if it doesn't disconnect the structure
            for i in range(len(child.positions)):
                temp_positions = [pos for j, pos in enumerate(child.positions) if j != i]

                # Quick connectivity check using BFS
                if len(temp_positions) <= 1:
                    is_connected = True
                else:
                    pos_set = set(tuple(round(coord, 6) for coord in pos) for pos in temp_positions)
                    start = tuple(round(coord, 6) for coord in temp_positions[0])
                    visited = {start}
                    queue = deque([start])

                    while queue:
                        current = queue.popleft()
                        for dx, dy, dz in directions:
                            neighbor = tuple(
                                round(coord, 6) for coord in [current[0] + dx, current[1] + dy, current[2] + dz])
                            if neighbor in pos_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                    is_connected = len(visited) == len(temp_positions)

                if is_connected:
                    child.positions.pop(i)
                    break
        else:
            # Add a voxel adjacent to existing structure
            if child.positions:
                current_positions = set(tuple(round(coord, 6) for coord in pos) for pos in child.positions)
                candidates = []
                for pos in child.positions:
                    for dx, dy, dz in directions:
                        new_pos = pos + np.array([dx, dy, dz])
                        new_pos_tuple = tuple(round(coord, 6) for coord in new_pos)
                        if new_pos_tuple not in current_positions:
                            candidates.append(new_pos)

                if candidates:
                    child.positions.append(random.choice(candidates))

    return child

# -----------------------------
# GA runners
# -----------------------------
def evaluate_population(population):
    fitnesses = []
    for ind in population:
        key = obj_to_key(ind)
        if key in EVAL_CACHE:
            ind.fitness = EVAL_CACHE[key]
        else:
            _, fit = AttemptGrips(ind, 0)
            ind.fitness = float(fit)
            update_cache(CACHE_FILE, ind, ind.fitness)
        fitnesses.append(ind.fitness)
    return np.array(fitnesses, dtype=float)

def run_standard_ga(generations=50, pop_size=10, n_parts=10, voxel_size=0.015, buffer=None):
    population = []
    for i in range(pop_size):
        dims = voxel_size * np.ones(3) / 2.0
        rgb = [(0.8, 0.2, 0.2)] * n_parts
        obj = SimObj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0,0,0])
        population.append(obj)
    best_history, mean_history = [], []
    for gen in range(generations):
        print("Generation",gen)
        fitnesses = evaluate_population(population)
        best_history.append(float(fitnesses.min()))
        mean_history.append(float(fitnesses.mean()))
        new_pop = []
        elite_count = max(2, int(pop_size * 0.15))
        idx_sorted = np.argsort(fitnesses)
        for i in range(elite_count): new_pop.append(population[idx_sorted[i]])
        while len(new_pop) < pop_size:
            def tournament_select(pop, fits, k=3):
                idxs = random.sample(range(len(pop)), min(k, len(pop)))
                chosen = idxs[np.argmin([fits[i] for i in idxs])]
                return pop[chosen]
            p1, p2 = tournament_select(population, fitnesses), tournament_select(population, fitnesses)
            child = voxel_crossover_mutation(p1, p2, voxel_size=voxel_size)
            _, fit = AttemptGrips(child, 0)
            child.fitness = float(fit)
            reward = compute_voxel_reward(child, p1, p2, population, fitnesses)
            if buffer is not None:
                state = encode_state_for_policy(p1, p2, population, fitnesses, k=n_parts)
                value_est = torch.tensor([0.0], device=device)
                buffer.push(state, 0, reward, torch.tensor(0.0), torch.tensor(0.0), value_est)
            new_pop.append(child)
        print("current population:",len(new_pop))
        population = new_pop
        if gen % 1 == 0:
            print(f"[StdGA Gen {gen}] Best={best_history[-1]:.2f} Mean={mean_history[-1]:.2f}")
    return population, best_history, mean_history
def run_standard_ga_with_actions(generations=5, pop_size=10, n_parts=10, voxel_size=0.015, max_open_positions=128, buffer=None):
    population = []
    for i in range(pop_size):
        dims = voxel_size * np.ones(3) / 2.0
        rgb = [(0.8, 0.2, 0.2)] * n_parts
        obj = SimObj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0,0,0])
        population.append(obj)
    best_history, mean_history = [], []
    for gen in range(generations):
        print("Generation",gen)
        fitnesses = evaluate_population(population)
        best_history.append(float(fitnesses.min()))
        mean_history.append(float(fitnesses.mean()))
        new_pop = []
        elite_count = max(2, int(pop_size * 0.15))
        idx_sorted = np.argsort(fitnesses)
        for i in range(elite_count): new_pop.append(population[idx_sorted[i]])
        while len(new_pop) < pop_size:
            def tournament_select(pop, fits, k=3):
                idxs = random.sample(range(len(pop)), min(k, len(pop)))
                chosen = idxs[np.argmin([fits[i] for i in idxs])]
                return pop[chosen]
            p1, p2 = tournament_select(population, fitnesses), tournament_select(population, fitnesses)
            child = voxel_crossover_mutation(p1, p2, voxel_size=voxel_size)
            _, fit = AttemptGrips(child, 0)
            child.fitness = float(fit)
            reward = compute_voxel_reward(child, p1, p2, population, fitnesses)
            if buffer is not None:
                state = encode_state_for_policy(p1, p2, population, fitnesses, k=n_parts)
                open_positions, valid_count = get_open_positions(p1.positions, voxel_size, max_open_positions)
                if valid_count > 0 and len(child.positions) > len(p1.positions):
                    added_voxel = child.positions[-1]
                    try:
                        action_idx = next(i for i, pos in enumerate(open_positions[:valid_count])if np.allclose(pos, added_voxel, atol=1e-6))
                    except StopIteration:
                        action_idx = 0
                else:
                    action_idx = 0
                value_est = torch.tensor([0.0], device=device)
                buffer.push(state, action_idx, reward,torch.tensor(0.0), torch.tensor(0.0), value_est)
            new_pop.append(child)
            population = new_pop
        if gen % 1 == 0:
            print(f"[StdGA-Act Gen {gen}] Best={best_history[-1]:.2f} Mean={mean_history[-1]:.2f}")
    return population, best_history, mean_history
def run_rl_guided_ga(generations, pop_size, n_parts, voxel_size, max_open_positions, policy, value_net, opt_policy, opt_value, buffer, rl_ratio=1.0):
    population = []
    for i in range(pop_size):
        dims = voxel_size * np.ones(3) / 2.0
        rgb = [(0.8, 0.2, 0.2)] * n_parts
        obj = SimObj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0,0,0])
        population.append(obj)
    best_history, mean_history = [], []
    for gen in range(generations):
        fitnesses = evaluate_population(population)
        best_history.append(float(fitnesses.min()))
        mean_history.append(float(fitnesses.mean()))
        new_pop = []
        elite_count = max(2, int(pop_size * 0.15))
        idx_sorted = np.argsort(fitnesses)
        for i in range(elite_count): new_pop.append(population[idx_sorted[i]])
        while len(new_pop) < pop_size:
            def tournament_select(pop, fits, k=3):
                idxs = random.sample(range(len(pop)), min(k, len(pop)))
                chosen = idxs[np.argmin([fits[i] for i in idxs])]
                return pop[chosen]
            p1, p2 = tournament_select(population, fitnesses), tournament_select(population, fitnesses)
            if random.random() < rl_ratio:
                open_positions, valid_count = get_open_positions(p1.positions, voxel_size, max_open_positions)
                if valid_count == 0:
                    child = voxel_crossover_mutation(p1, p2, voxel_size=voxel_size)
                else:
                    mask = torch.zeros((max_open_positions,), dtype=torch.bool, device=device)
                    mask[:valid_count] = True
                    state = encode_state_for_policy(p1, p2, population, fitnesses, k=n_parts)
                    logits = policy(state.unsqueeze(0))
                    logits_masked = logits.clone(); logits_masked[0, ~mask] = -1e9
                    dist = torch.distributions.Categorical(logits=logits_masked)
                    action_idx = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action_idx, device=device))
                    entropy = dist.entropy()
                    selected_pos = open_positions[action_idx]
                    child = copy.deepcopy(p1);
                    child.positions = child.positions[:-1]
                    child.positions.append(selected_pos)
                    if not is_connected(child.positions, voxel_size): child = copy.deepcopy(p1)
                    _, fit = AttemptGrips(child, 0); child.fitness = float(fit)
                    reward = compute_voxel_reward(child, p1, p2, population, fitnesses)
                    value_est = value_net(state.unsqueeze(0)).squeeze(0)
                    buffer.push(state, action_idx, reward, logp, entropy, value_est)
            else:
                child = voxel_crossover_mutation(p1, p2, voxel_size=voxel_size)
                _, fit = AttemptGrips(child, 0); child.fitness = float(fit)
                reward = compute_voxel_reward(child, p1, p2, population, fitnesses)
                state = encode_state_for_policy(p1, p2, population, fitnesses, k=n_parts)
                #buffer.push(state, 0, reward, torch.tensor(0.0), torch.tensor(0.0), torch.tensor([0.0], device=device))
            new_pop.append(child)
        population = new_pop
        if len(buffer) > 20:
            pol_loss, val_loss = ppo_update_categorical(policy, value_net, opt_policy, opt_value, buffer)
            if gen % 1 == 0:
                print(f"[RLGA Train {gen}] pol_loss={pol_loss:.4f}, val_loss={val_loss:.4f}")
        if gen % 1 == 0:
            print(f"[RLGA Gen {gen}] Best={best_history[-1]:.2f} Mean={mean_history[-1]:.2f} Buffer={len(buffer)}")
    return population, best_history, mean_history

# -----------------------------
import argparse
import random
import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from MultObjPick import Obj as SimObj
from BaselineLearner import AttemptGrips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--n_parts", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.015)
    parser.add_argument("--max_open_positions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    global EVAL_CACHE
    EVAL_CACHE = load_cache(CACHE_FILE)
    buffer = DiscreteReplayBuffer(capacity=200)


    k = args.n_parts; state_dim = k * 3 * 2 + 2 + 4
    policy = CategoricalPolicy(state_dim=state_dim, max_actions=args.max_open_positions).to(device)
    value_net = ValueNet(state_dim=state_dim).to(device)
    opt_policy = optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    opt_value = optim.Adam(value_net.parameters(), lr=3e-4, weight_decay=1e-5)

   # print("Training RL policy from collected buffer...")
    #for epoch in range(100):
        #pol_loss, val_loss = ppo_update_categorical(policy, value_net, opt_policy, opt_value, buffer)
        #print(f"[Pretrain Epoch {epoch}] pol_loss={pol_loss:.4f} val_loss={val_loss:.4f}")

    print("Running RL-guided GA phase (50 generations)...")
    _, rl_best, rl_mean = run_rl_guided_ga(generations=10, pop_size=args.pop_size, n_parts=args.n_parts,
                                          voxel_size=args.voxel_size, max_open_positions=args.max_open_positions,
                                          policy=policy, value_net=value_net,
                                          opt_policy=opt_policy, opt_value=opt_value,
                                          buffer=buffer, rl_ratio=0.7)
    print("RL-guided GA finished. Last 5 gens best:", rl_best[-5:], "mean:", rl_mean[-5:])

    print("Running standard GA phase (50 generations)...")
    _, std_best, std_mean = run_standard_ga_with_actions(generations=10, pop_size=args.pop_size, n_parts=args.n_parts,
                                           voxel_size=args.voxel_size, buffer=buffer)
    print("Standard GA finished. Last 5 gens best:", std_best[-5:], "mean:", std_mean[-5:])
    plt.figure()
    plt.plot(std_best, label="StdGA Best")
    plt.plot(std_mean, label="StdGA Mean")
    plt.plot(rl_best, label="RLGA Best")
    plt.plot(rl_mean, label="RLGA Mean")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("GA vs RL-Guided GA")
    plt.show()

if __name__ == "__main__":
    main()
