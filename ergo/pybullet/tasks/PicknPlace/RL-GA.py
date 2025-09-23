# rl_guided_ga_categorical_open_positions.py
"""
RL-guided GA for adversarial voxel object generation.

- Single RL mode: discrete categorical policy that selects from valid open neighbor positions.
- Evaluator: uses BaselineLearner.AttemptGrips (imported directly).
- GA: uses tournament selection + elitism + RL or GA offspring generation.
- PPO-like training for a categorical policy + value baseline.
"""

import argparse
import random
import copy
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# IMPORTANT: script expects these modules to be in PYTHONPATH (as you requested)
from MultObjPick import Obj as SimObj
from BaselineLearner import AttemptGrips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Utilities: open neighbor positions (ensures connectivity)
# -----------------------------
def get_open_positions(obj_positions, voxel_size=0.015, max_positions=128):
    """
    Given a list of np.array positions (voxels), return a list of candidate open neighbor
    positions (adjacent faces) up to max_positions. Returns a tuple (open_positions, mask_length)
    where open_positions is length == max_positions (padded with zeros), and mask_length is number
    of valid positions.
    """
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

    # deterministic ordering for reproducibility
    sorted_open = sorted(open_positions, key=lambda p: (p[0], p[1], p[2]))

    # truncate/pad
    valid_count = min(len(sorted_open), max_positions)
    padded = []
    for i in range(max_positions):
        if i < valid_count:
            padded.append(np.array(sorted_open[i], dtype=np.float32))
        else:
            padded.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return padded, valid_count

def is_connected(positions, voxel_size=0.015, tol=1e-6):
    """
    Quick BFS connectivity check on voxel positions (assumes axis-aligned grid).
    Positions: list of np.array
    """
    if len(positions) <= 1:
        return True
    coords = [tuple(np.round(p,6)) for p in positions]
    neighbors = set(coords)
    # adjacency offsets
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
# Policy (categorical) + Value net
# -----------------------------
class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, max_actions=128, hidden_dim=256):
        """
        Policy outputs logits over `max_actions`. The environment will mask invalid actions.
        state_dim: dimension of encoded state vector
        """
        super().__init__()
        self.max_actions = max_actions
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(hidden_dim, max_actions)  # logits over discrete positions

    def forward(self, state):
        """
        state: tensor shape (..., state_dim)
        returns logits (no softmax)
        """
        h = self.shared(state)
        logits = self.logits_head(h)
        return logits

    def get_dist(self, state, mask):
        """
        state: (batch, state_dim) or (state_dim,)
        mask: boolean mask shape (batch, max_actions) or (max_actions,) with True for valid actions
        returns: torch.distributions.Categorical (masked), log_probs for chosen action if provided
        """
        logits = self.forward(state)
        # apply mask by setting large negative value to invalid logits
        neg_inf = -1e9
        logits_masked = logits.clone()
        # mask shape handling
        logits_masked[~mask] = neg_inf
        dist = torch.distributions.Categorical(logits=logits_masked)
        return dist

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
# Replay buffer for discrete actions
# -----------------------------
class DiscreteReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, log_prob, entropy, value):
        """
        state: torch tensor (state_dim)
        action_idx: int or tensor scalar
        reward: float
        log_prob: tensor scalar
        entropy: tensor scalar
        value: tensor scalar
        """
        self.buffer.append((state.detach().cpu(), torch.tensor(action_idx).long(), float(reward),
                            log_prob.detach().cpu(), entropy.detach().cpu(), value.detach().cpu()))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, logps, ents, vals = zip(*batch)
        # stack tensors
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
# PPO-like update for categorical policy
# -----------------------------
def ppo_update_categorical(policy, value_net, opt_policy, opt_value, buffer,
                           batch_size=256, epochs=4, clip_epsilon=0.2, c1=0.5, c2=0.01):
    """
    Performs PPO style updates on minibatches sampled from buffer.
    c1: value loss coef, c2: entropy coef
    """
    if len(buffer) < batch_size:
        return 0.0, 0.0

    states, actions, rewards, old_logps, entropies, old_vals = buffer.sample(batch_size)

    # compute returns and advantages (simple, using reward as return here because episodes are 1-step-ish)
    # If you store multi-step returns, replace accordingly.
    returns = rewards
    with torch.no_grad():
        values = value_net(states)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss_total = 0.0
    value_loss_total = 0.0

    for _ in range(epochs):
        logits = policy(states)  # (B, max_actions)
        # we need masks: old_logps were computed with mask applied; we assume invalid logits extremely negative
        # Recompute dist and log probs for taken actions
        dist = torch.distributions.Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_logps - old_logps)
        # clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value_net(states), returns)

        loss = policy_loss + c1 * value_loss - c2 * entropy

        opt_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt_policy.step()

        opt_value.zero_grad()
        (value_loss * c1).backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        opt_value.step()

        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()

    return policy_loss_total / epochs, value_loss_total / epochs

# -----------------------------
# State encoder
# -----------------------------
def encode_state_for_policy(parent1, parent2, population_objs, population_fitnesses, k=10):
    """
    Encodes parents (top-k voxels around centroid), parent fitnesses, and population stats into a fixed vector.
    Returns torch.tensor of shape (state_dim,)
    """
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

    p1_enc = encode_obj(parent1, k)
    p2_enc = encode_obj(parent2, k)
    p1_fit = AttemptGrips(parent1, 0)[1]
    p2_fit = AttemptGrips(parent2, 0)[1]
    stats = [float(np.min(population_fitnesses)), float(np.max(population_fitnesses)),
             float(np.mean(population_fitnesses)), float(np.std(population_fitnesses))]
    state_list = p1_enc + p2_enc + [float(p1_fit), float(p2_fit)] + stats
    return torch.tensor(state_list, dtype=torch.float32, device=device)

# -----------------------------
# Voxel-level GA operators
# -----------------------------
def voxel_crossover_mutation(parent1, parent2, mutation_rate=0.2, voxel_size=0.015):
    child = copy.deepcopy(parent1)
    p2_positions = list(parent2.positions)
    random.shuffle(p2_positions)
    take = max(1, int(0.3 * len(p2_positions)))
    child.positions = list(child.positions) + p2_positions[:take]

    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(child.positions) > 1:
            idx = random.randrange(len(child.positions))
            child.positions.pop(idx)
        else:
            base = child.positions[random.randrange(len(child.positions))]
            dirs = [np.array([voxel_size,0,0]), np.array([-voxel_size,0,0]),
                    np.array([0,voxel_size,0]), np.array([0,-voxel_size,0]),
                    np.array([0,0,voxel_size]), np.array([0,0,-voxel_size])]
            child.positions.append(base + dirs[random.randrange(6)])
    return child

# -----------------------------
# Reward function (uses AttemptGrips directly)
# -----------------------------
def compute_voxel_reward(child_obj, parent1_obj, parent2_obj, population_objs, fitnesses):
    """
    Returns scalar reward for child. Lower graspability (fewer successful grasps) -> better.
    Uses same scheme as previously: improvement, ranking, diversity.
    """
    _, child_grasps = AttemptGrips(child_obj, 0)
    _, p1_grasps = AttemptGrips(parent1_obj, 0)
    _, p2_grasps = AttemptGrips(parent2_obj, 0)

    child_fit = float(child_grasps)
    best_parent = float(min(p1_grasps, p2_grasps)) + 1e-8
    improvement = (best_parent - child_fit) / (best_parent)
    better_than = sum(1 for f in fitnesses if child_fit < f)
    ranking = (better_than / max(1, len(fitnesses))) - 0.5

    # diversity: centroid distance
    def centroid(obj):
        return np.mean(np.array(obj.positions), axis=0)
    try:
        dists = [np.linalg.norm(centroid(child_obj) - centroid(p)) for p in population_objs]
        min_dist = float(np.min(dists)) if len(dists) > 0 else 0.0
    except Exception:
        min_dist = 0.0
    diversity = min(min_dist / 0.1, 1.0)

    total_reward = 2.0 * improvement + 1.0 * ranking + 0.5 * diversity
    return float(total_reward)

# -----------------------------
# RL-guided GA Runner
# -----------------------------
def run_rl_guided_ga(generations=100, pop_size=5, n_parts=12, voxel_size=0.015,
                     max_open_positions=128, rl_ratio=0.7, seed=0,
                     buffer_capacity=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # state dimension: encode 12 voxels per parent => 10*3*2 = 72, plus 2 parent fits + 4 stats = 78
    k = 12
    state_dim = k * 3 * 2 + 2 + 4
    policy = CategoricalPolicy(state_dim=state_dim, max_actions=max_open_positions, hidden_dim=256).to(device)
    value_net = ValueNet(state_dim=state_dim, hidden_dim=256).to(device)
    opt_policy = optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    opt_value = optim.Adam(value_net.parameters(), lr=3e-4, weight_decay=1e-5)
    buffer = DiscreteReplayBuffer(capacity=buffer_capacity)

    # initialize population
    population = []
    for i in range(pop_size):
        dims = voxel_size * np.ones(3) / 2.0
        rgb = [(0.8, 0.2, 0.2)] * n_parts  # tuple, not list
        obj = SimObj(dims, n_parts, rgb)
        # Use your generator API if available; otherwise assume GenerateObject exists
        obj.GenerateObject(dims, n_parts, [0,0,0])
        population.append(obj)

    best_history = []
    mean_history = []

    for gen in range(generations):
        # evaluate population using your AttemptGrips
        fitnesses = np.array([AttemptGrips(ind, 0)[1] for ind in population], dtype=float)
        best_history.append(float(fitnesses.min()))
        mean_history.append(float(fitnesses.mean()))

        # elitism
        new_pop = []
        elite_count = max(2, int(pop_size * 0.15))
        idx_sorted = np.argsort(fitnesses)
        for i in range(elite_count):
            new_pop.append(population[idx_sorted[i]])

        # generate rest of population
        while len(new_pop) < pop_size:
            # tournament selection
            def tournament_select(pop, fits, k=3):
                idxs = random.sample(range(len(pop)), min(k, len(pop)))
                chosen = idxs[np.argmin([fits[i] for i in idxs])]
                return pop[chosen]
            p1 = tournament_select(population, fitnesses, 3)
            p2 = tournament_select(population, fitnesses, 3)

            if random.random() < rl_ratio:
                # RL child: select one open position on parent1 (could use both parents in state)
                open_positions, valid_count = get_open_positions(p1.positions, voxel_size=voxel_size, max_positions=max_open_positions)
                if valid_count == 0:
                    # fallback to GA mutation if no open slots
                    child = voxel_crossover_mutation(p1, p2, mutation_rate=0.2, voxel_size=voxel_size)
                    new_pop.append(child)
                    continue

                # Build mask
                mask = torch.zeros((max_open_positions,), dtype=torch.bool, device=device)
                mask[:valid_count] = True

                # Build state
                state = encode_state_for_policy(p1, p2, population, fitnesses, k=k)

                # get distribution and sample action
                logits = policy(state.unsqueeze(0))  # (1, max_actions)
                # mask logits by setting invalid to large negative
                neg_inf = -1e9
                logits_masked = logits.clone()
                logits_masked[0, ~mask] = neg_inf
                dist = torch.distributions.Categorical(logits=logits_masked)
                action_idx = dist.sample().item()
                logp = dist.log_prob(torch.tensor(action_idx, device=device))
                entropy = dist.entropy()

                # create child by placing voxel at selected open position
                selected_pos = open_positions[action_idx]
                child = p1.clone_and_mutate_to(selected_pos) if hasattr(p1, 'clone_and_mutate_to') else copy.deepcopy(p1)
                # If clone_and_mutate_to not available, ensure we append manually
                if not hasattr(p1, 'clone_and_mutate_to'):
                    child.positions = list(copy.deepcopy(p1.positions))
                    child.positions.append(selected_pos)

                # ensure connectivity (should be connected by construction)
                if not is_connected(child.positions, voxel_size=voxel_size):
                    # If somehow disconnected, revert to parent (penalize later)
                    child = copy.deepcopy(p1)

                # compute reward and push to buffer
                reward = compute_voxel_reward(child, p1, p2, population, fitnesses)
                value_est = value_net(state.unsqueeze(0)).squeeze(0)
                buffer.push(state, action_idx, reward, logp, entropy, value_est)
                new_pop.append(child)
            else:
                # GA child
                child = voxel_crossover_mutation(p1, p2, mutation_rate=0.2, voxel_size=voxel_size)
                # make sure child is connected; if not, attempt repair by removing disjoint voxels
                if not is_connected(child.positions, voxel_size=voxel_size):
                    # simple repair: keep only largest connected component
                    coords = [tuple(np.round(p,6)) for p in child.positions]
                    neighbor_offsets = [(voxel_size,0,0),(-voxel_size,0,0),(0,voxel_size,0),(0,-voxel_size,0),(0,0,voxel_size),(0,0,-voxel_size)]
                    comp_sets = []
                    visited = set()
                    for i,c in enumerate(coords):
                        if c in visited:
                            continue
                        stack = [c]
                        comp = set([c])
                        visited.add(c)
                        while stack:
                            cur = stack.pop()
                            for off in neighbor_offsets:
                                nb = (round(cur[0]+off[0],6), round(cur[1]+off[1],6), round(cur[2]+off[2],6))
                                if nb in coords and nb not in visited:
                                    visited.add(nb)
                                    comp.add(nb)
                                    stack.append(nb)
                        comp_sets.append(comp)
                    # choose largest comp
                    largest = max(comp_sets, key=lambda s: len(s))
                    new_positions = [np.array(p) for p in largest]
                    child.positions = new_positions
                new_pop.append(child)

        population = new_pop

        # policy training
        if len(buffer) > 20:
            pol_loss, val_loss = ppo_update_categorical(policy, value_net, opt_policy, opt_value, buffer,
                                                       batch_size=256, epochs=4, clip_epsilon=0.2)
            if gen % 10 == 0:
                print(f"Training update at gen {gen}: pol_loss={pol_loss:.4f}, val_loss={val_loss:.4f}")

        if gen % 10 == 0:
            print(f"[Gen {gen}] Best={best_history[-1]:.2f} Mean={mean_history[-1]:.2f} Buffer={len(buffer)}")

    return population, best_history, mean_history

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--n_parts", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.015)
    parser.add_argument("--max_open_positions", type=int, default=128)
    parser.add_argument("--rl_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pop, bh, mh = run_rl_guided_ga(generations=args.generations,
                                   pop_size=args.pop_size,
                                   n_parts=args.n_parts,
                                   voxel_size=args.voxel_size,
                                   max_open_positions=args.max_open_positions,
                                   rl_ratio=args.rl_ratio,
                                   seed=args.seed)
    print("Finished run.")
    print("Best history (last 10):", bh[-10:])
    print("Mean history (last 10):", mh[-10:])

if __name__ == "__main__":
    main()
