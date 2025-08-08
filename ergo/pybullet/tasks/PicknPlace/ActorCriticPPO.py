import math
import os
import sys
import numpy as np
import pybullet as pb
import torch
import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch import distributions as dist
import pickle
import random

sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import table_position, table_half_extents
import MultObjPick
import pybullet as pb
import numpy as np




def get_object_part_world_positions(obj):
    """
    Converts the relative positions of object parts (voxels) to world coordinates.
    """
    base_pos, base_orn = pb.getBasePositionAndOrientation(obj.ObjId)
    rot_matrix = np.array(pb.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    world_positions = []
    for rel_pos in obj.positions:
        rel_pos_vec = np.array(rel_pos).reshape(3, 1)
        world_pos = np.array(base_pos).reshape(3, 1) + np.dot(rot_matrix, rel_pos_vec)
        world_positions.append(world_pos.flatten())
    return world_positions


def get_gripping_score(env, obj, gripper_link_indices, voxel_size,weight_centering=0.5, weight_alignment=0.5):
    """
    Calculates a single score representing how well the gripper is positioned to grasp the object.

    Args:
        env: The PyBullet environment object.
        obj: The object instance with voxel positions.
        gripper_link_indices: A list or tuple of the two gripper link indices.
        voxel_size (float): The size of each object voxel.
        weight_centering (float): The importance of the object being centered on the axis.
        weight_alignment (float): The importance of the object being between the gripper tips.

    Returns:
        A single `gripping_score` from 0 to 1, where 1 is a perfect gripping position.
    """
    # Get the world positions of the object's parts (voxels)
    object_part_positions = get_object_part_world_positions(obj)
    # Get the world positions of the gripper tips
    gripper_tip1_pos = np.array(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
    gripper_tip2_pos = np.array(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])
    gripper_midpoint_pos = (gripper_tip1_pos + gripper_tip2_pos) / 2.0
    # Define the gripper axis vector
    gripper_axis = gripper_tip2_pos - gripper_tip1_pos
    gripper_length = np.linalg.norm(gripper_axis)
    if gripper_length < 1e-6:
        return 0.0
    gripper_axis_normalized = gripper_axis / gripper_length
    # Use the passed-in voxel_size to set the buffer
    gripping_zone_buffer = 2 * voxel_size
    distances_to_axis = []
    alignment_scores = []
    for part_pos in object_part_positions:
        part_pos = np.array(part_pos)
        # Vector from the midpoint to the object part
        vec_to_part = part_pos - gripper_midpoint_pos
        # Projection onto the gripper axis (now a signed distance from the midpoint)
        projection_length = np.dot(vec_to_part, gripper_axis_normalized)
        # The projected point on the axis is now calculated from the midpoint
        projected_point = gripper_midpoint_pos + projection_length * gripper_axis_normalized
        distance_to_axis = np.linalg.norm(part_pos - projected_point)
        distances_to_axis.append(distance_to_axis)
        # Calculate alignment score for this specific part
        distance_penalty = np.exp(-100 * distance_to_axis ** 2)
        max_proj_length = gripper_length / 2.0 - gripping_zone_buffer
        if np.abs(projection_length) < max_proj_length:
            length_score = 1.0
        else:
            length_score = 0.0
        part_alignment_score = distance_penalty * length_score
        alignment_scores.append(part_alignment_score)
    mean_distance_to_axis = np.mean(distances_to_axis) if distances_to_axis else 1e6
    centering_score = np.exp(-50 * mean_distance_to_axis)
    alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0
    gripping_score = (centering_score * weight_centering) + (alignment_score * weight_alignment)

    return gripping_score


def get_orientation_score(env, obj, gripper_link_indices):
    """
    Calculates a score for the angular alignment between the object and the gripper.
    This function checks alignment with all three of the object's principal axes.

    Args:
        env: The PyBullet environment object.
        obj: The object instance.
        gripper_link_indices: A list or tuple of the two gripper link indices.

    Returns:
        A score from 0 to 1, where 1 is perfect alignment with any of the object's axes.
    """
    # Get the world positions of the gripper tips
    gripper_tip1_pos = np.array(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
    gripper_tip2_pos = np.array(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])

    # Calculate the normalized gripper axis vector
    gripper_axis = gripper_tip2_pos - gripper_tip1_pos
    gripper_axis_normalized = gripper_axis / np.linalg.norm(gripper_axis)

    # Get the object's orientation quaternion
    obj_pos, obj_orn = pb.getBasePositionAndOrientation(obj.ObjId)
    rot_matrix = np.array(pb.getMatrixFromQuaternion(obj_orn)).reshape(3, 3)

    # Extract the object's three principal axes (X, Y, and Z) in world coordinates
    object_axis_x = rot_matrix[:, 0]
    object_axis_y = rot_matrix[:, 1]
    object_axis_z = rot_matrix[:, 2]

    # Calculate the alignment score for each of the object's axes
    alignment_x = np.abs(np.dot(gripper_axis_normalized, object_axis_x))
    alignment_y = np.abs(np.dot(gripper_axis_normalized, object_axis_y))
    alignment_z = np.abs(np.dot(gripper_axis_normalized, object_axis_z))

    # The final orientation score is the maximum of the three alignments
    orientation_score = np.max([alignment_x, alignment_y, alignment_z])

    return orientation_score

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.approach_joints_idx = []
        self.pickup_joints_idx = []

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor(features)
        value = self.critic(features)
        log_std = self.log_std.clamp(-5, 0)
        std = torch.exp(log_std)
        return mean, std, value

    def get_action(self, state):
        mean, std, _ = self.forward(state)
        dist = tr.distributions.Normal(mean, std)
        action = dist.rsample()
        action_norm = torch.clamp(torch.tanh(action), -0.999, 0.999)
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1 - action_norm.pow(2) + 1e-6).sum(dim=-1)
        return action_norm, log_prob, std, action


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, clip_eps=0.2, epochs=10,
                 batch_size=500, minibatch_size=100):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.ema_alpha = 0.1
        self.last_approach_reward_avg = 0.0
        self.last_pickup_reward_avg = 0.0
        self.num_updates = 0

        # Batch storage
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_rewards = []
        self.batch_approach_rewards = []
        self.batch_pickup_rewards = []
        self.batch_distances = []
        self.batch_values = []
        self.batch_next_values = []
        self.batch_dones = []

    def store_transition(self, state, action, log_prob, reward, approach_reward,
                         pickup_reward, distance, value, next_value, done):
        """Store a single transition in the batch buffer"""
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_log_probs.append(log_prob)
        self.batch_rewards.append(reward)
        self.batch_approach_rewards.append(approach_reward)
        self.batch_pickup_rewards.append(pickup_reward)
        self.batch_distances.append(distance)
        self.batch_values.append(value)
        self.batch_next_values.append(next_value)
        self.batch_dones.append(done)

    def is_batch_ready(self):
        """Check if we have enough samples for a batch update"""
        return len(self.batch_states) >= self.batch_size

    def clear_batch(self):
        """Clear the batch buffer after update"""
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_log_probs.clear()
        self.batch_rewards.clear()
        self.batch_approach_rewards.clear()
        self.batch_pickup_rewards.clear()
        self.batch_distances.clear()
        self.batch_values.clear()
        self.batch_next_values.clear()
        self.batch_dones.clear()

    def compute_gae_advantages(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        device = values.device
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + 0.99 * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
        returns = torch.tensor(returns, device=device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def create_minibatches(self, states, actions, old_log_probs, rewards, approach_rewards,
                           pickup_rewards, distances, values, next_values, dones, advantages, returns):
        """Create minibatches for training"""
        batch_size = len(states)
        indices = list(range(batch_size))
        random.shuffle(indices)

        minibatches = []
        for start in range(0, batch_size, self.minibatch_size):
            end = min(start + self.minibatch_size, batch_size)
            mb_indices = indices[start:end]

            minibatch = {
                'states': states[mb_indices],
                'actions': actions[mb_indices],
                'old_log_probs': old_log_probs[mb_indices],
                'distances': distances[mb_indices],
                'values': values[mb_indices],
                'dones': dones[mb_indices],
                'advantages': advantages[mb_indices],
                'returns': returns[mb_indices]
            }
            minibatches.append(minibatch)

        return minibatches

    def update_batch(self):
        """Update using the collected batch with minibatch processing"""
        if not self.is_batch_ready():
            return

        device = next(self.model.parameters()).device

        # Convert batch data to tensors
        states = torch.stack(self.batch_states).to(device)
        actions = torch.stack(self.batch_actions).to(device)
        old_log_probs = torch.stack(self.batch_log_probs).to(device)
        rewards = torch.tensor(self.batch_rewards, dtype=torch.float32, device=device)
        approach_rewards = torch.tensor(self.batch_approach_rewards, dtype=torch.float32, device=device)
        pickup_rewards = torch.tensor(self.batch_pickup_rewards, dtype=torch.float32, device=device)
        distances = torch.tensor(self.batch_distances, dtype=torch.float32, device=device)
        values = torch.stack(self.batch_values).squeeze().to(device)
        next_values = torch.stack(self.batch_next_values).squeeze().to(device)
        dones = torch.tensor(self.batch_dones, dtype=torch.float32, device=device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae_advantages(rewards, values, next_values, dones)

        # Detach to avoid backward-through-graph error
        advantages = advantages.detach()
        returns = returns.detach()
        old_log_probs = old_log_probs.detach()
        states = states.detach()
        actions = actions.detach()

        # Update EMA rewards
        avg_approach_reward = torch.mean(approach_rewards).item()
        avg_pickup_reward = torch.mean(pickup_rewards).item()
        self.update_avg_rewards(avg_approach_reward, avg_pickup_reward)

        # Training loop with minibatches
        for epoch in range(self.epochs):
            minibatches = self.create_minibatches(
                states, actions, old_log_probs, rewards, approach_rewards,
                pickup_rewards, distances, values, next_values, dones, advantages, returns
            )

            for minibatch in minibatches:
                self.update_minibatch_spo(minibatch, device)

        # Clear the batch after update
        self.clear_batch()

    def update_minibatch_spo(self, minibatch, device):
        """Update using a single minibatch with SPO policy loss"""
        mb_states = minibatch['states']
        mb_actions = minibatch['actions']
        mb_old_log_probs = minibatch['old_log_probs']
        mb_advantages = minibatch['advantages']
        mb_returns = minibatch['returns']
        mb_dones = minibatch['dones']
        mb_values = minibatch['values']

        # Forward pass
        mean, std, new_values = self.model(mb_states)
        dist = torch.distributions.Normal(mean, std)

        # Tanh squash correction for log probs
        squashed_actions = torch.tanh(mb_actions)
        safe_actions = torch.clamp(squashed_actions, -0.99, 0.99)

        new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
        new_log_probs -= torch.log(1 - safe_actions.pow(2) + 1e-6).sum(dim=-1)

        # Calculate ratio
        ratio = torch.exp(new_log_probs - mb_old_log_probs)

        epsilon = self.clip_eps  # reuse clip_eps as epsilon for SPO

        # SPO policy loss (quadratic penalty)
        policy_loss_elements = ratio * mb_advantages - (mb_advantages.abs() / (2 * epsilon)) * (ratio - 1).pow(2)
        policy_loss = -policy_loss_elements.mean()

        # Entropy bonus (same as your PPO)
        assert len(self.model.approach_joints_idx) > 0, "Approach joints not set"

        joint_entropies = dist.entropy()
        normalized_approach = torch.clamp(torch.tensor(self.last_approach_reward_avg), -1.0, 3.0)
        normalized_approach_pos = (normalized_approach + 1.0) / 4.0
        normalized_pickup = torch.clamp(torch.tensor(self.last_pickup_reward_avg), 0.0, 1.0)
        inv_approach = 1.0 - normalized_approach_pos
        inv_pickup = 1.0 - normalized_pickup
        inv_pickup = inv_pickup.to(device)
        inv_approach = inv_approach.to(device)

        min_beta = 0.001
        max_beta = 0.01
        denominator = math.expm1(1)
        frac = torch.expm1(inv_approach) / math.expm1(2)
        current_approach_entropy_coeff = min_beta + (max_beta - min_beta) * frac
        frac_p = torch.expm1(inv_pickup) / denominator
        current_pickup_entropy_coeff = min_beta + (max_beta - min_beta) * frac_p

        dones_mask = (1 - mb_dones).float().to(device)
        entropy_approach = joint_entropies[:, self.model.approach_joints_idx].sum(dim=-1) * dones_mask
        entropy_pickup = joint_entropies[:, self.model.pickup_joints_idx].sum(dim=-1) * dones_mask

        total_entropy_bonus = (current_approach_entropy_coeff * entropy_approach +
                               current_pickup_entropy_coeff * entropy_pickup).mean()

        policy_loss = policy_loss - total_entropy_bonus

        # Value loss (same clipped loss as PPO)
        value_pred_clipped = mb_values + torch.clamp(new_values.squeeze() - mb_values,-epsilon, epsilon)
        value_loss1 = (new_values.squeeze() - mb_returns).pow(2)
        value_loss2 = (value_pred_clipped - mb_returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def update_minibatch(self, minibatch, device):
        """Update using a single minibatch"""
        mb_states = minibatch['states']
        mb_actions = minibatch['actions']
        mb_old_log_probs = minibatch['old_log_probs']
        mb_distances = minibatch['distances']
        mb_advantages = minibatch['advantages']
        mb_returns = minibatch['returns']
        mb_dones = minibatch['dones']
        mb_values = minibatch['values']

        # Forward pass
        mean, std, new_values = self.model(mb_states)
        dist = tr.distributions.Normal(mean, std)

        # Convert actions to normalized version using tanh
        squashed_actions = torch.tanh(mb_actions)
        safe_actions = torch.clamp(squashed_actions, -0.99, 0.99)

        new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
        new_log_probs -= torch.log(1 - safe_actions.pow(2) + 1e-6).sum(dim=-1)

        # Policy loss
        ratio = torch.exp(new_log_probs - mb_old_log_probs)
        surrogate1 = ratio * mb_advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Entropy bonus
        assert len(self.model.approach_joints_idx) > 0, "Approach joints not set"

        joint_entropies = dist.entropy()
        normalized_approach = torch.clamp(torch.tensor(self.last_approach_reward_avg), -1.0, 3.0)
        normalized_approach_pos = (normalized_approach + 1.0) / 4.0
        normalized_pickup = torch.clamp(torch.tensor(self.last_pickup_reward_avg), 0.0, 1.0)
        inv_approach = 1.0 - normalized_approach_pos
        inv_pickup = 1.0 - normalized_pickup
        inv_pickup = inv_pickup.to(device)
        inv_approach = inv_approach.to(device)

        min_beta = 0.001
        max_beta = 0.01
        denominator = math.expm1(1)
        frac = torch.expm1(inv_approach) / math.expm1(2)
        current_approach_entropy_coeff = min_beta + (max_beta - min_beta) * frac
        frac_p = torch.expm1(inv_pickup) / denominator
        current_pickup_entropy_coeff = min_beta + (max_beta - min_beta) * frac_p

        dones_mask = (1 - mb_dones).float().to(device)
        entropy_approach = joint_entropies[:, self.model.approach_joints_idx].sum(dim=-1) * dones_mask
        entropy_pickup = joint_entropies[:, self.model.pickup_joints_idx].sum(dim=-1) * dones_mask

        total_entropy_bonus = (current_approach_entropy_coeff * entropy_approach +
                               current_pickup_entropy_coeff * entropy_pickup).mean()

        policy_loss = policy_loss - total_entropy_bonus

        # Value loss
        value_pred_clipped = mb_values + torch.clamp(new_values.squeeze() - mb_values,
                                                     -self.clip_eps, self.clip_eps)
        value_loss1 = (new_values.squeeze() - mb_returns).pow(2)
        value_loss2 = (value_pred_clipped - mb_returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def update_avg_rewards_perstep(self, current_approach_reward, current_pickup_reward):
        if len(self.last_approach_reward_avg) < 1:
            self.last_approach_reward_avg = current_approach_reward
        if len(self.last_pickup_reward_avg) < 1:
            self.last_pickup_reward_avg = current_pickup_reward
            return
        if len(self.last_pickup_reward_avg) >= 1 and len(self.last_approach_reward_avg) >= 1:
            self.last_approach_reward_avg = (
                    (1 - self.ema_alpha) * self.last_approach_reward_avg + self.ema_alpha * current_approach_reward
            )
            self.last_pickup_reward_avg = (
                    (1 - self.ema_alpha) * self.last_pickup_reward_avg + self.ema_alpha * current_pickup_reward
            )

    def update_avg_rewards(self, episode_approach_reward, episode_pickup_reward):
        """
        Update EMA rewards with single episode values instead of batch averages

        Args:
            episode_approach_reward: Single float value for approach reward from one episode
            episode_pickup_reward: Single float value for pickup reward from one episode
        """
        # Convert to float if tensor
        if hasattr(episode_approach_reward, 'item'):
            episode_approach_reward = episode_approach_reward.item()
        if hasattr(episode_pickup_reward, 'item'):
            episode_pickup_reward = episode_pickup_reward.item()

        # Initialize if first episode
        if self.last_approach_reward_avg == 0.0:
            self.last_approach_reward_avg = episode_approach_reward
            self.last_pickup_reward_avg = episode_pickup_reward
            return

        self.last_approach_reward_avg = (
                    (1 - self.ema_alpha) * self.last_approach_reward_avg + self.ema_alpha * episode_approach_reward
            )
        self.last_pickup_reward_avg = (
                    (1 - self.ema_alpha) * self.last_pickup_reward_avg + self.ema_alpha * episode_pickup_reward
            )

def make_state(angles, pos1, pos2, object_pos, obj_orientation, use_right_hand, obj_part_positions):
    objp = tr.tensor(object_pos)
    objo = tr.tensor(obj_orientation)
    rangles = tr.tensor(angles)
    use_right_hand_value = int(use_right_hand)
    use_right_hand_tensor = tr.tensor([use_right_hand_value], dtype=tr.float32)
    base_state = tr.cat((rangles, pos1, pos2, objp, objo, use_right_hand_tensor))

    if obj_part_positions is not None:
        part_tensor = torch.tensor(np.array(obj_part_positions), dtype=tr.float32).flatten()
        return torch.cat((part_tensor, base_state))
    else:
        return base_state


old_obj_height = 0.0
was_touching = False

def new_rewards(env,obj,obj_pos,obj_id,gripper_link_indices,v_f):
    ip_positions = [torch.tensor(pb.getLinkState(env.robot_id, idx)[0]) for idx in gripper_link_indices]

    tip_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id,linkIndexA=gripper_link_indices[0], linkIndexB=gripper_link_indices[1])
    tip_distance = torch.norm(ip_positions[0] - ip_positions[1])
    gripper_midpoint = (ip_positions[0] + ip_positions[1]) / 2.0
    v_f_np = np.array(v_f)
    v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)
    distances = torch.norm(v_f_tensor - gripper_midpoint, dim=1)


    mean_distance = torch.mean(distances)
    approach_reward = torch.tensor(0.0)
    approach_reward -= 4 * mean_distance
    approach_reward += get_gripping_score(env,obj,gripper_link_indices,0.015) #[0,1]
    approach_reward += 0.1*get_orientation_score(env,obj,gripper_link_indices) #[0,1]       # Range [0,2] - mean stuff

    # Check if any voxel is within gripper bounding box
    # if False , ideal width 2 x voxel
    # if true , slightly lower than voxel

    # give rewards for picking up the object and touching

    gripper_reward = torch.tensor(0.0)
    voxels_in_gripper = 0
    for voxel_pos in v_f_tensor:
        # Project voxel onto gripper axis to check if it's between the tips
        tip1_to_voxel = voxel_pos - ip_positions[0]
        tip1_to_tip2 = ip_positions[1] - ip_positions[0]

        # Check if voxel is roughly between the gripper tips
        projection = torch.dot(tip1_to_voxel, tip1_to_tip2) / torch.norm(tip1_to_tip2)
        if 0 <= projection <= torch.norm(tip1_to_tip2):
            distance_to_line = torch.norm(tip1_to_voxel - projection * tip1_to_tip2 / torch.norm(tip1_to_tip2))
            if distance_to_line < 0.02:  # Voxel is close to gripper line
                voxels_in_gripper += 1
    if voxels_in_gripper>0:
        print("Object Voxel betweem gripper!")
    # Reward based on gripper width appropriateness
    if voxels_in_gripper > 0:
        # Object between grippers - reward slightly closed gripper
        optimal_width = 0.013  # Slightly less than voxel size
        width_error = abs(tip_distance - optimal_width)
        gripper_reward += 0.3 * torch.exp(-10 * width_error)
    else:
        # No object between grippers - reward wider gripper (2x voxel size)
        optimal_width = 0.03  # 2x voxel size for approach
        width_error = abs(tip_distance - optimal_width)
        gripper_reward += 0.1 * torch.exp(-5 * width_error)

    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
    height_diff = obj_height - obj_pos[2]

    fingers_touching = 0
    for link_idx in gripper_link_indices:
        contact_points = pb.getContactPoints(
            bodyA=env.robot_id,
            bodyB=obj_id,
            linkIndexA=link_idx,
            linkIndexB=-1
        )
        if len(contact_points) > 0:
            fingers_touching += 1

    if fingers_touching == 1:
        gripper_reward += 0.1
    is_touching = False
    if fingers_touching >= 2:
        gripper_reward += 1
        is_touching = True
        if was_touching == True:
            gripper_reward += 0.5
            print("Continuous touch reward")
        else:
            print("first double touch reward ")

    if pb.getBasePositionAndOrientation(obj_id)[0][0] > obj_pos[0] +0.3:
        print("moved object ")
        approach_reward-=1.0
    height_reward = torch.tensor(0.0)
    g_check = False
    if height_diff > 0.01 and is_touching:
        if height_diff < 0.05:
            height_reward = 2.0 * (height_diff / 0.05)
        else:
            height_reward = 2.0 + 1.0 * (height_diff - 0.05)
    if obj_height > obj_pos[2] + 0.2 and is_touching:
        g_check = True
        approach_reward += 10.0
        print("Pick success")
    approach_reward += height_reward
    total_reward = approach_reward+gripper_reward
    return total_reward,approach_reward,gripper_reward,g_check

def rewards_potential(env, obj_pos, obj_id, gripper_link_indices, v_f, old_distance, scale_factor):
    g_check = False
    global old_obj_height
    global was_touching
    ip_positions = [torch.tensor(pb.getLinkState(env.robot_id, idx)[0]) for idx in gripper_link_indices]

    tip_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id,
                                       linkIndexA=gripper_link_indices[0], linkIndexB=gripper_link_indices[1])
    tip_distance = torch.norm(ip_positions[0] - ip_positions[1])
    gripper_midpoint = (ip_positions[0] + ip_positions[1]) / 2.0

    # Compute distance to nearest voxel
    v_f_np = np.array(v_f)
    v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)
    distances = torch.norm(v_f_tensor - gripper_midpoint, dim=1)
    new_distance = torch.min(distances)
    mean_distance = torch.mean(distances)

    # Compute approach reward as potential difference
    if old_distance is None:
        approach_reward = torch.tensor(0.0)
    else:
        diff = (old_distance - mean_distance) * scale_factor
        diff_clamped = torch.clamp(diff, -1.0, 1.0)
        approach_reward = diff_clamped
    if new_distance < 0.05:
        approach_reward += 1

    if new_distance <= 0.05:
        max_dist = 0.05
        norm_dist = (max_dist - new_distance) / max_dist
        approach_reward = approach_reward + (norm_dist ** 2)
    else:
        approach_reward -= 0.1

    gripper_width_reward = torch.tensor(0.0)
    optimal_width = 0.018
    width_tolerance = 0.006
    width_error = torch.abs(tip_distance - optimal_width)
    if width_error < width_tolerance:
        gripper_width_reward = 0.5 * (1.0 - width_error / width_tolerance)
    else:
        excess_error = width_error - width_tolerance
        gripper_width_reward = -0.1 * torch.clamp(excess_error / width_tolerance, 0.0, 2.0)

    pickup_reward = torch.tensor(0.0)
    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
    height_reward = torch.tensor(0.0)
    height_diff = obj_height - obj_pos[2]
    fingers_touching = 0

    for link_idx in gripper_link_indices:
        contact_points = pb.getContactPoints(
            bodyA=env.robot_id,
            bodyB=obj_id,
            linkIndexA=link_idx,
            linkIndexB=-1
        )
        if len(contact_points) > 0:
            fingers_touching += 1

    if fingers_touching == 1:
        pickup_reward += 0.1

    is_touching = False
    if fingers_touching >= 2:
        pickup_reward += 0.5
        is_touching = True
        if was_touching == True:
            pickup_reward += 0.5
            print("Continuous touch reward")
        else:
            print("first double touch reward ")

    if is_touching == False and was_touching == True:
        pickup_reward -= 0.5
        approach_reward -= 0.1
        print("Stopped touching penality")

    if height_diff > 0.01 and is_touching:
        if height_diff < 0.05:
            height_reward = 2.0 * (height_diff / 0.05)
        else:
            height_reward = 2.0 + 1.0 * (height_diff - 0.05)

    pickup_reward += height_reward
    pickup_reward += gripper_width_reward

    if obj_height > obj_pos[2] + 0.2 and is_touching:
        g_check = True
        pickup_reward += 10.0
        print("Pick success")

    if len(tip_contacts) > 0:
        pickup_reward -= 0.01

    if obj_height - old_obj_height < -0.1 and was_touching == True and is_touching == False:
        print("Dropping object penality!")
        pickup_reward -= 1.0

    was_touching = is_touching
    total_reward = approach_reward + pickup_reward

    return approach_reward, pickup_reward, g_check, new_distance, total_reward


def get_joint_limits(robot_id, joint_indices):
    """Returns a dictionary mapping joint index to (lower, upper) limits."""
    joint_limits = {}
    for idx in joint_indices:
        info = pb.getJointInfo(robot_id, idx)
        joint_limits[idx] = (info[8], info[9])
    return joint_limits


def unnormalize_actions(normalized_action, joint_indices, joint_limits):
    """Converts normalized actions in [-1, 1] to absolute joint angles for given joint indices."""
    unnormalized = []
    for i, idx in enumerate(joint_indices):
        low, high = joint_limits[idx]
        scaled = (normalized_action[i] + 1.0) * 0.5 * (high - low) + low
        unnormalized.append(scaled)
    return unnormalized


def append_rewards_to_pickle(new_rewards, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_rewards = pickle.load(f)
    else:
        all_rewards = []
    all_rewards.extend(new_rewards)
    with open(file_path, 'wb') as f:
        pickle.dump(all_rewards, f)


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Current device index:", torch.cuda.current_device())
    else:
        print("CUDA not available. 😕")

    state_dim = 86
    action_dim = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent with batch collection parameters
    agent = PPOAgent(state_dim, action_dim, batch_size=500, minibatch_size=100)
    agent.model.to(device)

    # Tracking variables
    rewards_out = []
    areward_out = []
    preward_out = []
    log_p = []
    std_d = []
    resume = 0
    episode = 0
    max_episodes = 1000000
    voxel_size = 0.015
    dims = voxel_size * np.ones(3) / 2
    n_parts = 10
    rgb = [(.75, .25, .25)] * n_parts
    obj = MultObjPick.Obj(dims, n_parts, rgb)
    obj.GenerateObject(dims, n_parts, [0, 0, 0])

    if resume == 1:
        checkpoint = torch.load('ppo_checkpoint_6.pth', map_location=device)
        print(checkpoint.keys())
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['epoch'] + 1
        obj.positions = checkpoint['object']
        print("Loaded checkpoint")

    # Main training loop
    while episode < max_episodes:
        table_height = table_position()[2] + table_half_extents()[2]
        show = False

        if episode % 10000 == 0:
            show = False

        exp = MultObjPick.experiment(show)
        exp.CreateScene()
        env = exp.env

        pb.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=-1.2, cameraPitch=-39.0,
                                      cameraTargetPosition=(0., 0., 0.))
        obj_id = exp.Spawn_Object(obj)
        state_angles = env.get_position()

        if episode != 0:
            exp.env.settle(exp.env.get_position(), seconds=0.2)
        else:
            exp.env.settle(exp.env.get_position(), seconds=3)

        obj_pos, obj_orientation = pb.getBasePositionAndOrientation(obj_id)
        if obj_pos[2] < table_height:
            pb.removeBody(obj_id)
            print("Obj fell off on episode, restarting", episode)
            pb.removeBody(obj.ObjId)
            env.close()
            obj = MultObjPick.Obj(dims, n_parts, rgb)
            obj.GenerateObject(dims, n_parts, [0, 0, 0])
            continue

        # Choose hand based on distance
        left_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        right_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
        dist_left = np.linalg.norm(left_hand_pos - obj_pos)
        dist_right = np.linalg.norm(right_hand_pos - obj_pos)
        use_right_hand = dist_right < dist_left

        if use_right_hand:
            arm_indices = [32, 33, 34, 35, 36, 37, 38]
            pickup_joints = [38]
            approach_joints = [32, 33, 34, 35, 36, 37]
            gripper_link_indices = [env.joint_index["r_moving_tip"], env.joint_index["r_fixed_tip"]]
        else:
            arm_indices = [22, 23, 24, 25, 26, 27, 28]
            pickup_joints = [28]
            approach_joints = [22, 23, 24, 25, 26, 27]
            gripper_link_indices = [env.joint_index["l_moving_tip"], env.joint_index["l_fixed_tip"]]

        pos1 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
        pos2 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])

        # Set collision filters
        pb.setCollisionFilterPair(env.robot_id, env.robot_id, gripper_link_indices[0], gripper_link_indices[1],
                                  enableCollision=True)
        pb.setCollisionFilterPair(env.robot_id, env.robot_id, gripper_link_indices[0], pickup_joints[0],
                                  enableCollision=True)
        pb.setCollisionFilterPair(env.robot_id, env.robot_id, gripper_link_indices[1], pickup_joints[0],
                                  enableCollision=True)
        pb.setCollisionFilterPair(env.robot_id, env.robot_id, pickup_joints[0], approach_joints[-1],
                                  enableCollision=True)

        joint_index_map = {joint_idx: i for i, joint_idx in enumerate(arm_indices)}
        agent.model.pickup_joints_idx = [joint_index_map[j] for j in pickup_joints]
        agent.model.approach_joints_idx = [joint_index_map[j] for j in approach_joints]
        joint_limits = get_joint_limits(env.robot_id, arm_indices)

        v_f = get_object_part_world_positions(obj)
        midpoint = (pos1 + pos2) / 2.0
        v_f_np = np.array(v_f)
        v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)
        distances = torch.norm(v_f_tensor - midpoint, dim=1)
        old_distance = torch.min(distances)
        state = make_state(state_angles, pos1, pos2, obj_pos, obj_orientation, use_right_hand, v_f).to(device)

        done = False
        steps = 0
        episode_rewards = []
        episode_approach_rewards = []
        episode_pickup_rewards = []
        was_touching = False

        # Episode loop - collect transitions for batch
        while steps < 10:  # Changed from <= to <
            state_tensor = state.float()
            action, log_prob, std_dev, raw_action = agent.model.get_action(state_tensor)
            next_angles = env.get_position()
            old_obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
            action_unnorm = unnormalize_actions(action, arm_indices, joint_limits)
            for i, idx in enumerate(arm_indices):
                next_angles[idx] = action_unnorm[i]

            env.goto_position(list(next_angles))
            next_state_angles = env.get_position()
            next_obj_pos, next_obj_orientation = pb.getBasePositionAndOrientation(obj_id)
            pos1 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
            pos2 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])
            v_f = get_object_part_world_positions(obj)
            next_state = make_state(next_state_angles, pos1, pos2, next_obj_pos, next_obj_orientation, use_right_hand,v_f).to(device)
            total_reward,areward,preward,graspcheck = new_rewards(env,obj, obj_pos, obj_id, gripper_link_indices, v_f)
            distances = torch.norm(v_f_tensor - midpoint, dim=1)
            #old_distance = torch.min(distances)
            new_distance = torch.min(distances)
            reward = total_reward
            reward = reward - 0.1 * steps  # Using current step count for penalty
            # Early termination if grasp successful
            done = graspcheck
            # Add bonus reward for early success (before incrementing steps)
            if graspcheck == True and steps < 10:
                reward += (10 - steps)  # -1 because we're about to increment
            steps = steps + 1
            # Episode ends at max steps or early success
            if steps > 10:
                done = True
            # Get current value
            with torch.no_grad():
                current_value = agent.model(state_tensor)[2].detach().squeeze()

            # Get next value (or final value if done)
            if done:

                next_value = torch.tensor(0.0, device=device)
            else:
                with torch.no_grad():
                    next_value = agent.model(next_state.float())[2].detach().squeeze()

            # Store transition in batch buffer
            agent.store_transition(
                state=state_tensor,
                action=raw_action,
                log_prob=log_prob,
                reward=reward.item() if hasattr(reward, 'item') else reward,
                approach_reward=areward.item() if hasattr(areward, 'item') else areward,
                pickup_reward=preward.item() if hasattr(preward, 'item') else preward,
                distance=new_distance.item() if hasattr(new_distance, 'item') else new_distance,
                value=current_value,
                next_value=next_value,
                done=float(done)
            )

            # Track episode statistics
            episode_rewards.append(reward.item() if hasattr(reward, 'item') else reward)
            episode_approach_rewards.append(areward.item() if hasattr(areward, 'item') else areward)
            episode_pickup_rewards.append(preward.item() if hasattr(preward, 'item') else preward)

            state = next_state

            if done:
                break
        agent.update_avg_rewards(sum(episode_approach_rewards), sum(episode_pickup_rewards))
        # Update batch if ready
        if agent.is_batch_ready():
            print(f"Updating batch at episode {episode}")
            agent.update_batch()
            agent.num_updates += 1
            if agent.num_updates % 10 == 0:
                print(f"Episode {episode}, Tot_reward: {sum(episode_rewards):.3f}, "
                      f"Avg.ApproachReward: {agent.last_pickup_reward_avg:.3f}, "
                      f"Avg.Pickreward: {agent.last_pickup_reward_avg:.3f}, "
                      )
            if agent.num_updates % 10 == 0:
                print(f"Episode {episode}, StdDev: {[sdev.item() for sdev in std_dev]}")
        # Store episode statistics
        rewards_out.append(sum(episode_rewards))
        log_p.append(tr.mean(torch.tensor([log_prob])) if hasattr(log_prob, 'item') else log_prob)
        std_d.append(std_dev)
        areward_out.append(sum(episode_approach_rewards))
        preward_out.append(sum(episode_pickup_rewards))

        # Save data periodically
        if episode % 1000 == 0 and episode != 0:
            append_rewards_to_pickle(log_p, 'log_probs2.pickle')
            log_p.clear()

            append_rewards_to_pickle(std_d, 'std_dev2.pickle')
            std_d.clear()

            append_rewards_to_pickle(rewards_out, 'rewards2.pickle')
            print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            rewards_out.clear()

            # Save checkpoint
            torch.save({
                'epoch': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'object': obj.positions,
                'batch_size': len(agent.batch_states)
            }, 'ppo_checkpoint_6.pth')

        env.close()
        episode += 1

    # Final batch update if there are remaining samples
    if len(agent.batch_states) > 0:
        print("Final batch update with remaining samples")
        if len(agent.batch_states) >= agent.minibatch_size:
            agent.batch_size = len(agent.batch_states)
            agent.update_batch()

    print("Training completed!")