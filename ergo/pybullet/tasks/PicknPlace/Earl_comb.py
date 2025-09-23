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
# Adversarial A2C Policy
# -----------------------------
class AdversarialA2CPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Action heads
        self.split_head = nn.Linear(hidden_dim // 2, 10)
        self.combine_head = nn.Linear(hidden_dim // 2, 6)
        self.mutate_head = nn.Linear(hidden_dim // 2, 5)
        self.intensity_head = nn.Linear(hidden_dim // 2, 1)

        # Value head
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state):
        features = self.shared(state)
        return {
            'split_logits': self.split_head(features),
            'combine_logits': self.combine_head(features),
            'mutate_logits': self.mutate_head(features),
            'intensity': torch.sigmoid(self.intensity_head(features)),
            'value': self.value_head(features)
        }

    def act(self, state):
        with torch.no_grad():
            output = self.forward(state.unsqueeze(0))

            split_dist = torch.distributions.Categorical(logits=output['split_logits'])
            combine_dist = torch.distributions.Categorical(logits=output['combine_logits'])
            mutate_dist = torch.distributions.Categorical(logits=output['mutate_logits'])

            split1_action = split_dist.sample()
            split2_action = split_dist.sample()
            combine_action = combine_dist.sample()
            mutate_action = mutate_dist.sample()
            intensity = output['intensity'].squeeze()

            total_logp = (split_dist.log_prob(split1_action) + split_dist.log_prob(split2_action) +
                          combine_dist.log_prob(combine_action) + mutate_dist.log_prob(mutate_action))
            entropy = split_dist.entropy() * 2 + combine_dist.entropy() + mutate_dist.entropy()
            value = output['value'].squeeze()

            actions = {
                'split': [split1_action.item(), split2_action.item()],
                'combine': combine_action.item(),
                'mutate': mutate_action.item(),
                'intensity': intensity.item()
            }

        return actions, total_logp, entropy, value


class AdversarialReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, log_prob, entropy, value):
        self.buffer.append({
            'state': state.cpu(),
            'actions': actions,
            'reward': reward,
            'log_prob': log_prob.cpu(),
            'entropy': entropy.cpu(),
            'value': value.cpu()
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.stack([item['state'] for item in batch]).to(device)
        actions = [item['actions'] for item in batch]
        rewards = torch.tensor([item['reward'] for item in batch], device=device)
        log_probs = torch.stack([item['log_prob'] for item in batch]).to(device)
        entropies = torch.stack([item['entropy'] for item in batch]).to(device)
        values = torch.stack([item['value'] for item in batch]).to(device)
        return states, actions, rewards, log_probs, entropies, values

    def __len__(self):
        return len(self.buffer)


def adversarial_crossover_mutation(parent1, parent2, actions, voxel_size=0.015):
    """Adversarial crossover using policy actions"""

    def split_parent(parent, split_action):
        positions = list(parent.positions)
        if len(positions) <= 1:
            return positions, []

        if split_action == 0:  # Random half
            random.shuffle(positions)
            mid = len(positions) // 2
            return positions[:mid], positions[mid:]
        elif split_action == 1:  # X-axis
            positions.sort(key=lambda p: p[0])
            mid = len(positions) // 2
            return positions[:mid], positions[mid:]
        elif split_action == 2:  # Y-axis
            positions.sort(key=lambda p: p[1])
            mid = len(positions) // 2
            return positions[:mid], positions[mid:]
        elif split_action == 3:  # Z-axis
            positions.sort(key=lambda p: p[2])
            mid = len(positions) // 2
            return positions[:mid], positions[mid:]
        elif split_action == 4:  # Core vs peripheral
            if len(positions) < 3:
                random.shuffle(positions)
                mid = len(positions) // 2
                return positions[:mid], positions[mid:]
            centroid = np.mean(positions, axis=0)
            distances = [(i, np.linalg.norm(p - centroid)) for i, p in enumerate(positions)]
            distances.sort(key=lambda x: x[1])
            mid = len(distances) // 2
            core = [positions[distances[i][0]] for i in range(mid)]
            peripheral = [positions[distances[i][0]] for i in range(mid, len(distances))]
            return core, peripheral
        else:  # Ratio splits (5-9)
            ratios = [0.3, 0.7, 0.25, 0.75, 0.4]
            ratio = ratios[split_action - 5]
            random.shuffle(positions)
            split_point = max(1, int(len(positions) * ratio))
            return positions[:split_point], positions[split_point:]

    def combine_parts(p1_part1, p1_part2, p2_part1, p2_part2, combine_action):
        all_parts = [p for p in [p1_part1, p1_part2, p2_part1, p2_part2] if p]
        if not all_parts:
            return list(parent1.positions)

        if combine_action == 0:
            return (p1_part1 if p1_part1 else []) + (p2_part1 if p2_part1 else [])
        elif combine_action == 1:
            return (p1_part1 if p1_part1 else []) + (p2_part2 if p2_part2 else [])
        elif combine_action == 2:
            return (p1_part2 if p1_part2 else []) + (p2_part1 if p2_part1 else [])
        elif combine_action == 3:
            return (p1_part2 if p1_part2 else []) + (p2_part2 if p2_part2 else [])
        elif combine_action == 4:  # Interleave
            combined = []
            max_len = max(len(part) for part in all_parts)
            for i in range(max_len):
                for part in all_parts:
                    if i < len(part):
                        combined.append(part[i])
            return combined
        else:  # Random selection
            all_voxels = []
            for part in all_parts:
                all_voxels.extend(part)
            if not all_voxels:
                return list(parent1.positions)
            random.shuffle(all_voxels)
            take_count = random.randint(max(1, len(all_voxels) // 3), len(all_voxels))
            return all_voxels[:take_count]

    def apply_mutation(positions, mutate_action, intensity):
        if not positions or intensity < 0.1:
            return positions

        positions = list(positions)
        num_mutations = max(1, int(len(positions) * intensity * 0.3))

        directions = [
            np.array([voxel_size, 0, 0]), np.array([-voxel_size, 0, 0]),
            np.array([0, voxel_size, 0]), np.array([0, -voxel_size, 0]),
            np.array([0, 0, voxel_size]), np.array([0, 0, -voxel_size])
        ]

        for _ in range(num_mutations):
            if mutate_action == 0:  # Add adjacent
                if positions:
                    base = random.choice(positions)
                    new_pos = base + random.choice(directions)
                    positions.append(new_pos)
            elif mutate_action == 1:  # Remove
                if len(positions) > 1:
                    positions.pop(random.randint(0, len(positions) - 1))
            elif mutate_action == 2:  # Move voxel
                if positions:
                    idx = random.randint(0, len(positions) - 1)
                    positions[idx] = positions[idx] + random.choice(directions)
            elif mutate_action == 3:  # Add cluster
                if positions:
                    base = random.choice(positions)
                    for _ in range(random.randint(2, 4)):
                        base = base + random.choice(directions)
                        positions.append(base)
            else:  # Add noise
                for i in range(len(positions)):
                    if random.random() < 0.2:
                        noise = np.random.normal(0, voxel_size * 0.05, 3)
                        positions[i] = positions[i] + noise

        return positions

    try:
        p1_part1, p1_part2 = split_parent(parent1, actions['split'][0])
        p2_part1, p2_part2 = split_parent(parent2, actions['split'][1])
        combined = combine_parts(p1_part1, p1_part2, p2_part1, p2_part2, actions['combine'])

        if combined:
            final_positions = apply_mutation(combined, actions['mutate'], actions['intensity'])
        else:
            final_positions = list(parent1.positions)

        child = copy.deepcopy(parent1)
        child.positions = final_positions if final_positions else list(parent1.positions)
        return child

    except Exception:
        # Fallback
        child = copy.deepcopy(parent1)
        p1_pos = list(parent1.positions)
        p2_pos = list(parent2.positions)
        random.shuffle(p1_pos)
        random.shuffle(p2_pos)
        child.positions = p1_pos[:len(p1_pos) // 2] + p2_pos[:len(p2_pos) // 2]
        return child


def compute_adversarial_reward(child_obj, parent1_obj, parent2_obj, pop_objs, fitnesses):
    """Adversarial reward focusing on diversity and novelty"""
    child_fit = float(child_obj.fitness)

    # 1. Fitness component
    if len(fitnesses) > 0:
        max_fit, min_fit = float(np.max(fitnesses)), float(np.min(fitnesses))
        if max_fit != min_fit:
            fitness_norm = (max_fit - child_fit) / (max_fit - min_fit)
            fitness_reward = max(0.0, fitness_norm - 0.2)
        else:
            fitness_reward = 0.5
    else:
        fitness_reward = 0.5

    # 2. Diversity reward
    def get_centroid(obj):
        if not obj.positions:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(np.array(obj.positions), axis=0)

    child_centroid = get_centroid(child_obj)
    diversity_score = 0.0

    if pop_objs:
        distances = []
        for pop_obj in pop_objs:
            pop_centroid = get_centroid(pop_obj)
            dist = np.linalg.norm(child_centroid - pop_centroid)
            distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            diversity_score = min(1.0, avg_distance / 0.1)

    # 3. Size diversity
    child_size = len(child_obj.positions)
    size_diversity = 0.0

    if pop_objs:
        pop_sizes = [len(obj.positions) for obj in pop_objs]
        if np.std(pop_sizes) > 0:
            size_diff = abs(child_size - np.mean(pop_sizes))
            size_diversity = min(1.0, size_diff / (np.std(pop_sizes) + 1e-8))

    # 4. Parent difference
    parent_diff = 0.0
    try:
        p1_centroid = get_centroid(parent1_obj)
        p2_centroid = get_centroid(parent2_obj)
        p1_dist = np.linalg.norm(child_centroid - p1_centroid)
        p2_dist = np.linalg.norm(child_centroid - p2_centroid)
        parent_diff = min(1.0, (p1_dist + p2_dist) / 0.2)
    except:
        parent_diff = 0.0

    return float(0.2 * fitness_reward + 0.4 * diversity_score + 0.2 * size_diversity + 0.2 * parent_diff)


def ppo_update_adversarial(policy, optimizer, buffer, batch_size=32, epochs=4,
                           clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, gamma=0.99):
    """PPO update for adversarial policy"""
    if len(buffer) < batch_size:
        return 0.0, 0.0

    states, actions_batch, rewards, old_log_probs, old_entropies, old_values = buffer.sample(batch_size)
    returns = rewards  # Simple case without next values
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss, total_value_loss = 0.0, 0.0

    for epoch in range(epochs):
        outputs = policy(states)

        new_log_probs = []
        new_entropies = []

        for i, actions in enumerate(actions_batch):
            split_dist = torch.distributions.Categorical(logits=outputs['split_logits'][i:i + 1])
            combine_dist = torch.distributions.Categorical(logits=outputs['combine_logits'][i:i + 1])
            mutate_dist = torch.distributions.Categorical(logits=outputs['mutate_logits'][i:i + 1])

            split1_logp = split_dist.log_prob(torch.tensor([actions['split'][0]], device=device))
            split2_logp = split_dist.log_prob(torch.tensor([actions['split'][1]], device=device))
            combine_logp = combine_dist.log_prob(torch.tensor([actions['combine']], device=device))
            mutate_logp = mutate_dist.log_prob(torch.tensor([actions['mutate']], device=device))

            total_logp = split1_logp + split2_logp + combine_logp + mutate_logp
            entropy = split_dist.entropy() * 2 + combine_dist.entropy() + mutate_dist.entropy()

            new_log_probs.append(total_logp.squeeze())
            new_entropies.append(entropy.squeeze())

        new_log_probs = torch.stack(new_log_probs)
        new_entropies = torch.stack(new_entropies)
        new_values = outputs['value'].squeeze()

        # PPO losses
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -new_entropies.mean()

        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / epochs, total_value_loss / epochs


# -----------------------------
# Existing utilities (from your code)
# -----------------------------
CACHE_FILE = "evaluated_objects.csv"
EVAL_CACHE = {}


def obj_to_key(obj):
    coords = [tuple(np.round(p, 6)) for p in obj.positions]
    coords_sorted = tuple(sorted(coords))
    return str(coords_sorted)


def load_cache(filename=CACHE_FILE):
    cache = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                cache[row[0]] = float(row[1])
    return cache


def update_cache(filename, obj, fitness):
    key = obj_to_key(obj)
    if key not in EVAL_CACHE:
        EVAL_CACHE[key] = fitness
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["object_key", "fitness"])
            writer.writerow([key, fitness])


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
    stats = [float(np.min(fitnesses)), float(np.max(fitnesses)),
             float(np.mean(fitnesses)), float(np.std(fitnesses))]
    state_list = p1_enc + p2_enc + [float(p1_fit), float(p2_fit)] + stats
    return torch.tensor(state_list, dtype=torch.float32, device=device)


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


def run_adversarial_ga(generations=50, pop_size=10, n_parts=10, voxel_size=0.015):
    """Run adversarial GA experiment"""

    # Initialize population
    population = []
    for i in range(pop_size):
        dims = voxel_size * np.ones(3) / 2.0
        rgb = [(0.8, 0.2, 0.2)] * n_parts
        obj = SimObj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        population.append(obj)

    # Initialize adversarial policy
    k = n_parts
    state_dim = k * 3 * 2 + 2 + 4  # Same as your encoding
    adversarial_policy = AdversarialA2CPolicy(state_dim).to(device)
    adversarial_optimizer = optim.Adam(adversarial_policy.parameters(), lr=3e-4)
    adversarial_buffer = AdversarialReplayBuffer(capacity=1000)

    best_history, mean_history = [], []

    for gen in range(generations):
        print(f"Generation {gen}")
        fitnesses = evaluate_population(population)
        best_history.append(float(fitnesses.min()))
        mean_history.append(float(fitnesses.mean()))

        new_pop = []
        elite_count = max(2, int(pop_size * 0.15))
        idx_sorted = np.argsort(fitnesses)

        # Keep elites
        for i in range(elite_count):
            new_pop.append(population[idx_sorted[i]])

        # Generate new individuals
        while len(new_pop) < pop_size:
            # Tournament selection
            def tournament_select(pop, fits, k=3):
                idxs = random.sample(range(len(pop)), min(k, len(pop)))
                chosen = idxs[np.argmin([fits[i] for i in idxs])]
                return pop[chosen]

            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)

            # Create state and sample adversarial action
            state = encode_state_for_policy(p1, p2, population, fitnesses, k=k)
            actions, log_prob, entropy, value = adversarial_policy.act(state)

            # Create child using adversarial crossover
            child = adversarial_crossover_mutation(p1, p2, actions, voxel_size)

            # Evaluate child
            _, fit = AttemptGrips(child, 0)
            child.fitness = float(fit)

            # Compute adversarial reward
            reward = compute_adversarial_reward(child, p1, p2, population, fitnesses)

            # Store experience
            adversarial_buffer.push(state, actions, reward, log_prob, entropy, value)

            new_pop.append(child)

        population = new_pop

        # Train adversarial policy
        if len(adversarial_buffer) > 32:
            pol_loss, val_loss = ppo_update_adversarial(adversarial_policy, adversarial_optimizer,
                                                        adversarial_buffer, batch_size=32, epochs=2)
            if gen % 1 == 0:
                print(f"[Adversarial Train {gen}] pol_loss={pol_loss:.4f}, val_loss={val_loss:.4f}")

        if gen % 1 == 0:
            print(f"[Adversarial GA Gen {gen}] Best={best_history[-1]:.2f} Mean={mean_history[-1]:.2f}")

    return population, best_history, mean_history, adversarial_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--n_parts", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.015)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    global EVAL_CACHE
    EVAL_CACHE = load_cache(CACHE_FILE)

    print("Running Adversarial GA...")
    final_pop, best_hist, mean_hist, policy = run_adversarial_ga(
        generations=args.generations,
        pop_size=args.pop_size,
        n_parts=args.n_parts,
        voxel_size=args.voxel_size
    )

    print("Adversarial GA finished.")
    print(f"Final best fitness: {best_hist[-1]:.3f}")
    print(f"Final mean fitness: {mean_hist[-1]:.3f}")

    # Plot results
    plt.figure()
    plt.plot(best_hist, label="Adversarial GA Best", color='red')
    plt.plot(mean_hist, label="Adversarial GA Mean", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Adversarial GA Performance")
    plt.show()


if __name__ == "__main__":
    main()