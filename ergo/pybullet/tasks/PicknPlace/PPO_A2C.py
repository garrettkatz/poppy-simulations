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

sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
#from ergo import PoppyErgoEnv
#import tabletop as tt
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
import BaselineLearner
from operator import add
import platform

def get_object_part_world_positions(obj):
    base_pos, base_orn = pb.getBasePositionAndOrientation(obj.ObjId)
    rot_matrix = np.array(pb.getMatrixFromQuaternion(base_orn)).reshape(3,3)
    world_positions = []
    for rel_pos in obj.positions:
        rel_pos_vec = np.array(rel_pos).reshape(3,1)
        world_pos = np.array(base_pos).reshape(3,1) + np.dot(rot_matrix, rel_pos_vec)
        world_positions.append(world_pos.flatten())
    return world_positions


def transform_voxels_to_gripper_frame(voxel_positions, gripper_pos, gripper_orn):

    rot_matrix = np.array(pb.getMatrixFromQuaternion(gripper_orn)).reshape(3,3)
    gripper_pos = np.array(gripper_pos)

    voxels_local = []
    for voxel in voxel_positions:
        vec = np.array(voxel) - gripper_pos
        local_point = rot_matrix.T @ vec
        voxels_local.append(local_point)

    return np.array(voxels_local)
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
        self.log_std = nn.Parameter(torch.ones(action_dim) * (-1.0))  # exp(-1.5) ≈ 0.22 radians

        self.approach_joints_idx = []
        self.pickup_joints_idx = []

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor(features)
        value = self.critic(features)
        std = torch.clamp(torch.exp(self.log_std), min=0.01, max=0.7)
        return mean, std, value

    def get_action(self, state):
        mean, std, _ = self.forward(state)
       # mean = torch.tanh(mean)
        dist = tr.distributions.Normal(mean, std)
        action = dist.rsample()
        action_norm = torch.clamp(torch.tanh(action), -0.999, 0.999)
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1 - action_norm.pow(2) + 1e-6).sum(dim=-1)
        return action_norm, log_prob,std,action

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, clip_eps=0.2, epochs=5):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.ema_alpha = 0.1
        self.last_approach_reward_avg =[]
        self.last_pickup_reward_avg = []
        #self.approach_joints_idx = []
        #self.pickup_joints_idx = []
        self.ema = 0.1
    def compute_advantage(self, rewards, values, next_values, dones):
        advantages, returns = [], []
        advantage = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * 0.95 * advantage
            returns.insert(0, advantage + values[i])
            advantages.insert(0, advantage)
        device = values.device
        return tr.tensor(advantages,device=device), tr.tensor(returns,device=device)

    def compute_gae_advantages(self, rewards, values,next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        device = values.device
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
        returns = torch.tensor(returns, device=device, dtype=torch.float32)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, approach_rewards, pickup_rewards,distance, values, next_values, dones):
        advantages, returns = self.compute_advantage(rewards, values, next_values, dones)

        # FIX: Detach to avoid backward-through-graph error
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        advantages = advantages.detach()
        returns = returns.detach()
        old_log_probs = old_log_probs.detach()
        states = states.detach()
        #distance = torch.tensor(distance, device=states.device)
        actions = actions.detach()
        dones_mask = (1 - dones).float().to(device)  # 1 for non-terminal, 0 for terminal

        for _ in range(self.epochs):
            mean, std, new_values = self.model(states)
            dist = tr.distributions.Normal(mean, std)

            # Unsquash actions using atanh (inverse tanh)
            #eps = 1e-6
            #raw_actions = 0.5 * torch.log((1 + actions) / (1 - actions + 1e-8))
            #safe_actions = torch.clamp(actions, -0.99, 0.99)
            squashed_actions = torch.tanh(actions)
            safe_actions = torch.clamp(squashed_actions, -0.99, 0.99)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            new_log_probs -= torch.log(1 - safe_actions.pow(2) + 1e-6).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            assert len(self.model.approach_joints_idx) > 0, "Approach joints not set"


            joint_entropies = dist.entropy()
           # normalized_approach = torch.tensor(approach_rewards).sum(dim=-1).mean()
            normalized_approach = torch.clamp(torch.tensor(self.last_approach_reward_avg), -1.0, 1.0)
            normalized_approach_pos = (normalized_approach + 1.0) / 2.0
            normalized_pickup = torch.clamp(torch.tensor(self.last_pickup_reward_avg), 0.0, 1.0)
            inv_approach = 1.0 - normalized_approach_pos
            inv_pickup = 1.0 - normalized_pickup
            inv_pickup = inv_pickup.to(device)
            inv_approach = inv_approach.to(device)
            pickup_mask= (distance<=0.03).float().to(values.device)
            pickup_mask_inverse =(distance>0.03).float().to(values.device)
            min_beta = 0.001
            max_beta = 0.01
            denominator = math.expm1(1)
            frac = torch.expm1(inv_approach)/denominator
            current_approach_entropy_coeff = min_beta +(max_beta-min_beta)*frac
            frac_p = torch.expm1(inv_pickup) / denominator

            current_pickup_entropy_coeff = (pickup_mask * (min_beta + (max_beta - min_beta) * frac_p) + pickup_mask_inverse * min_beta)

            entropy_approach = joint_entropies[:, self.model.approach_joints_idx].sum(dim=-1) * dones_mask
            entropy_pickup = joint_entropies[:, self.model.pickup_joints_idx].sum(dim=-1) * dones_mask

            total_entropy_bonus = (current_approach_entropy_coeff * entropy_approach + current_pickup_entropy_coeff * entropy_pickup).mean()

            policy_loss = policy_loss - total_entropy_bonus

            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

    def update_avg_rewards(self, current_approach_reward, current_pickup_reward):
        if len(self.last_approach_reward_avg)<1:
            self.last_approach_reward_avg = current_approach_reward
        if len(self.last_pickup_reward_avg) < 1:
            self.last_pickup_reward_avg = current_pickup_reward
            return
        if len(self.last_pickup_reward_avg)>= 1 and len(self.last_approach_reward_avg)>=1:
            self.last_approach_reward_avg = (
                    (1 - self.ema_alpha) * self.last_approach_reward_avg + self.ema_alpha * current_approach_reward
            )
            self.last_pickup_reward_avg = (
                    (1 - self.ema_alpha) * self.last_pickup_reward_avg + self.ema_alpha * current_pickup_reward
            )

def make_state(angles, object_pos, obj_orientation,use_right_hand,obj_part_positions,pos1,pos2):
    objp = tr.tensor(object_pos)
    objo = tr.tensor(obj_orientation)
    rangles = tr.tensor(angles)
    tip1pos = tr.tensor(pos1)
    tip2pos = tr.tensor(pos2)
    use_right_hand_value = int(use_right_hand)  # converts True → 1, False → 0
    use_right_hand_tensor = tr.tensor([use_right_hand_value], dtype=tr.float32)
    # Base state (15D): joint angles + object position + orientation + hand indicator
    base_state = tr.cat((rangles,tip1pos,tip2pos, objp, objo, use_right_hand_tensor))

    # Optionally concatenate voxel-based object features
    if obj_part_positions is not None:
        part_tensor = torch.tensor(np.array(obj_part_positions), dtype=tr.float32).flatten()
        return torch.cat((part_tensor,base_state))
    else:
        return base_state

old_obj_height = 0.0
def rewards_potential(env, obj_pos, obj_id, gripper_link_indices, v_f, old_distance=None, scale_factor=1.0):
    g_check = False

    ip_positions = [torch.tensor(pb.getLinkState(env.robot_id, idx)[0]) for idx in gripper_link_indices]

    tip_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id,
                                       linkIndexA=gripper_link_indices[0], linkIndexB=gripper_link_indices[1])
    tip_distance = torch.norm(ip_positions[0] - ip_positions[1])

    gripper_midpoint = (ip_positions[0] + ip_positions[1]) / 2.0

    # Compute distance to nearest voxel
    v_f_np = np.array(v_f)
    v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)  # shape (n_voxels, 3)
    distances = torch.norm(v_f_tensor - gripper_midpoint, dim=1)
    new_distance = torch.min(distances)

    # → Compute approach reward as potential difference
    proximity_bonus_max =1.0
    prox_thres = 0.05
    PROXIMITY_BONUS_SCALE = proximity_bonus_max / (prox_thres)
    if old_distance is None:
        approach_reward = torch.tensor(0.0)
    else:
        diff = (old_distance - new_distance)*scale_factor #0.045
        diff_clamped = torch.clamp(diff, -1.0, 1.0)
        approach_reward = diff_clamped
    if new_distance < prox_thres:
        prox_bonus = (prox_thres - new_distance) * PROXIMITY_BONUS_SCALE
        prox_bonus = torch.clamp(prox_bonus,0.0,proximity_bonus_max)
        approach_reward+=prox_bonus


    optimal_width = 0.015
    width_tolerance = 0.012
    width_error = torch.abs(tip_distance - optimal_width)
    if width_error < width_tolerance:
        gripper_width_reward = 0.5 * (1.0 - width_error / width_tolerance)  # Max 0.5
    else:
        excess_error = width_error - width_tolerance
        gripper_width_reward = -0.3 * torch.clamp(excess_error / width_tolerance, 0.0, 2.0)


    pickup_reward = torch.tensor(0.0)
    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
    height_reward = torch.tensor(0.0)
    height_diff = obj_height - obj_pos[2]

    if height_diff > 0.01:
        if height_diff < 0.05:
            height_reward = 2.0 * (height_diff / 0.05)  # Max 2.0 for initial lift
        else:
            height_reward = 2.0 + 1.0 * (height_diff - 0.05)  # +1.0 per additional 0.05

    pickup_reward += height_reward
    pickup_reward += gripper_width_reward


    all_contact_points = []
    fingers_touching = 0
    for link_idx in gripper_link_indices:
        contact_points = pb.getContactPoints(bodyA=env.robot_id,bodyB=obj_id,linkIndexA=link_idx,linkIndexB=-1)
        all_contact_points.append(contact_points)
        if len(contact_points) > 0:  # This finger is touching
            fingers_touching += 1
    if fingers_touching == 1:
        pickup_reward += 0.5
    if fingers_touching >= 2:
        pickup_reward += 1

    if obj_height > obj_pos[2] + 0.2:
        g_check = True
        pickup_reward+=10.0
        print("Pick success")

#penalities
    if len(tip_contacts) > 0:
        pickup_reward -= 0.3
        # Penalty for object falling
    if obj_height - old_obj_height  < -0.01:
        print("Dropping object penality!")
        pickup_reward -=1.0

    total_reward = approach_reward + pickup_reward
    return approach_reward, pickup_reward, g_check, new_distance, total_reward

def is_voxel_between_gripper_collision_boxes_single_function(robot_id, gripper_link_indices, voxel_pos):

    if len(gripper_link_indices) != 2:
        raise ValueError("gripper_link_indices must contain exactly two link IDs for the tips.")

    voxel_pos = np.array(voxel_pos)

    try:
        aabb1_min_tuple, aabb1_max_tuple = pb.getAABB(robot_id, gripper_link_indices[0])
        aabb1_min = np.array(aabb1_min_tuple)
        aabb1_max = np.array(aabb1_max_tuple)
    except pb.error:
        return False

    try:
        aabb2_min_tuple, aabb2_max_tuple = pb.getAABB(robot_id, gripper_link_indices[1])
        aabb2_min = np.array(aabb2_min_tuple)
        aabb2_max = np.array(aabb2_max_tuple)
    except pb.error:
        return False

    combined_min_x = min(aabb1_min[0], aabb2_min[0])
    combined_max_x = max(aabb1_max[0], aabb2_max[0])
    combined_min_y = min(aabb1_min[1], aabb2_min[1])
    combined_max_y = max(aabb1_max[1], aabb2_max[1])
    combined_min_z = min(aabb1_min[2], aabb2_min[2])
    combined_max_z = max(aabb1_max[2], aabb2_max[2])

    is_in_x = combined_min_x <= voxel_pos[0] <= combined_max_x
    is_in_y = combined_min_y <= voxel_pos[1] <= combined_max_y
    is_in_z = combined_min_z <= voxel_pos[2] <= combined_max_z

    return is_in_x and is_in_y and is_in_z

def check_pincer_alignment_improved_weighted(robot_id, ip_positions, v_f, gripper_link_indices,
                                  max_tip_dist_from_voxel=0.015 * 2,
                                  min_tip_separation=0.01,
                                  max_tip_separation=0.05,
                                  alignment_tolerance=0.015 * 0.5,
                                  weight_axial=0.5,          # New parameter: weight for axial alignment
                                  weight_perpendicular=0.5,  # New parameter: weight for perpendicular alignment
                                  epsilon=1e-6):

    if len(ip_positions) != 2:
        raise ValueError("ip_positions must contain exactly two gripper tip positions.")

    tip1, tip2 = ip_positions[0], ip_positions[1]
    best_alignment_score = -1.0
    best_voxel = None

    tip_separation = torch.norm(tip1 - tip2)
    total_norm_align_score =0.0
    for voxel_pos in v_f:
        voxel_tensor = torch.tensor(voxel_pos, dtype=torch.float32)

        if not is_voxel_between_gripper_collision_boxes_single_function(robot_id, gripper_link_indices, voxel_pos):
            continue

        dist1 = torch.norm(tip1 - voxel_tensor)
        dist2 = torch.norm(tip2 - voxel_tensor)

        if not (dist1 < max_tip_dist_from_voxel and dist2 < max_tip_dist_from_voxel):
            continue

        midpoint = (tip1 + tip2) / 2
        voxel_to_midpoint_vec = voxel_tensor - midpoint
        gripper_axis_vec = tip2 - tip1

        if tip_separation**2 < epsilon:
            projection_onto_gripper_axis = torch.zeros_like(gripper_axis_vec)
        else:
            projection_onto_gripper_axis = torch.dot(voxel_to_midpoint_vec, gripper_axis_vec) / (tip_separation**2) * gripper_axis_vec

        perpendicular_component = voxel_to_midpoint_vec - projection_onto_gripper_axis
        perpendicular_dist = torch.norm(perpendicular_component)
        axial_dist_from_midpoint = torch.norm(projection_onto_gripper_axis)

        axial_score = 1.0 - torch.clamp(axial_dist_from_midpoint / (tip_separation / 2 + epsilon), 0.0, 1.0)
        perpendicular_score = 1.0 - torch.clamp(perpendicular_dist / (tip_separation / 2 + alignment_tolerance + epsilon), 0.0, 1.0)


        # Calculate a weighted average of the two scores
        current_alignment_score = (weight_axial * axial_score + weight_perpendicular * perpendicular_score) / (weight_axial + weight_perpendicular + epsilon) # Add epsilon to denominator for safety

        total_norm_align_score+=current_alignment_score
    norm_align_score = total_norm_align_score/len(v_f)
    return norm_align_score

def check_pincer_alignment(ip_positions, v_f, tolerance=0.01):  # Reduce tolerance
    tip1, tip2 = ip_positions[0], ip_positions[1]
    best_alignment_score = 0
    best_voxel = None

    for voxel_pos in v_f:
        voxel_tensor = torch.tensor(voxel_pos)
        dist1 = torch.norm(tip1 - voxel_tensor)
        dist2 = torch.norm(tip2 - voxel_tensor)

        midpoint = (tip1 + tip2) / 2
        voxel_to_midpoint_dist = torch.norm(voxel_tensor - midpoint)
        tip_separation = torch.norm(tip1 - tip2)

        # Adjusted for 0.015 voxel size:
        if (dist1 < 0.045 and dist2 < 0.045 and  # ~3x voxel size
                voxel_to_midpoint_dist < tip_separation / 2 + tolerance and
                tip_separation < 0.04):  # Reasonable gripper closing range

            alignment_score = 1.0 - (voxel_to_midpoint_dist / (tip_separation / 2))
            if alignment_score > best_alignment_score:
                best_alignment_score = alignment_score
                best_voxel = voxel_pos

    return best_alignment_score, best_voxel


def get_joint_limits(robot_id, joint_indices):
    """
    Returns a dictionary mapping joint index to (lower, upper) limits.
    Only includes the specified joint indices.
    """
    joint_limits = {}
    for idx in joint_indices:
        info = pb.getJointInfo(robot_id, idx)
        joint_limits[idx] = (info[8], info[9])  # (lower_limit, upper_limit)
    return joint_limits

def get_object_voxels(voxels):
    flattened = torch.tensor(voxels, dtype=torch.float32).flatten()
    return flattened

def unnormalize_actions(normalized_action, joint_indices, joint_limits):
    """
    Converts normalized actions in [-1, 1] to absolute joint angles for given joint indices.
    Returns a list of joint angle values (same length as joint_indices).
    """
    unnormalized = []
    for i, idx in enumerate(joint_indices):
        low, high = joint_limits[idx]
        # Scale from [-1, 1] → [low, high]
        scaled = (normalized_action[i] + 1.0) * 0.5 * (high - low) + low
        unnormalized.append(scaled)
    return unnormalized

PICKLE_FILE = 'rewards10thexperiment_contactP_entropy_rewards.pickle'
PICKLE_FILE2 = 'rewards10thexperiment_contactP_entropy_logprob.pickle'
PICKLE_FILE3 = 'rewards10thdexperiment_contactP_entropy_stddev.pickle'

PICKLE_FILE4 = 'rewards10thexperiment_contactP_entropy_arewards.pickle'

PICKLE_FILE5 = 'rewards10thexperiment_contactP_entropy_prewards.pickle'
# Function to append rewards to the pickle file
def append_rewards_to_pickle(new_rewards, file_path):
    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_rewards = pickle.load(f)
    else:
        all_rewards = []

    # Append new rewards
    all_rewards.extend(new_rewards)

    # Save updated data
    with open(file_path, 'wb') as f:
        pickle.dump(all_rewards, f)


import MultObjPick
import pybullet as pb
if __name__ == "__main__":
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Current device index:", torch.cuda.current_device())
    else:
        print("CUDA not available. 😕")
    print(sys.executable)
    print(torch.__file__)

   # env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=False)
    state_dim = 86  # Adjusted for robot state representation
    action_dim = 7  # Assuming 8 joints as actions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(state_dim, action_dim)
    agent.model.to(device)
    rewards_out = []
    areward_out = []
    preward_out = []
    log_p = []
    std_d = []
    resume = 0
    episode = 0
    max_episodes = 1000000
    voxel_size = 0.015  # dimension of each voxel
    num_prll_prcs = 5
    dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
    n_parts = 10
    rgb = [(.75, .25, .25)] * n_parts
    gen0_results = []
    obj = MultObjPick.Obj(dims, n_parts, rgb)
    grasp_width = 1  # distance between grippers in voxel units
    obj.GenerateObject(dims, n_parts, [0, 0, 0])


    if resume== 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load('ppo_checkpoint.pth', map_location=device)
        print(checkpoint.keys())
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['epoch']
       # obj.positions = checkpoint['object']
        print("Loaded checkpoint")
   # print(f"Joint {i}: {info[1].decode('utf-8')}")
    while episode < max_episodes:

        table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

        Num_success = 0
        Num_Grips_attempted = 0
        Result = []
        num_objects = 1
       # obj.GenerateObject(dims, n_parts, [0, 0, 0])

        exp = MultObjPick.experiment()
        exp.CreateScene()
        env = exp.env

        pb.resetDebugVisualizerCamera(cameraDistance=1.4,cameraYaw=-1.2,cameraPitch=-39.0,cameraTargetPosition=(0., 0., 0.),)


        obj_id = exp.Spawn_Object(obj)


        state_angles = env.get_position()

        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        if episode !=0:

            exp.env.settle(exp.env.get_position(), seconds=0.2)
        else:
            exp.env.settle(exp.env.get_position(), seconds=3)

        obj_pos, obj_orientation = pb.getBasePositionAndOrientation(obj_id)

        if obj_pos[2] < table_height:
            pb.removeBody(obj_id)
            print("Obj fell off on episode,restarting",episode)
            pb.removeBody(obj.ObjId)
            env.close()
            obj = MultObjPick.Obj(dims, n_parts, rgb)
            obj.GenerateObject(dims, n_parts, [0, 0, 0])
            continue

        # pos of both hands
        left_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        right_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])

        # check distance and choose hand
        dist_left = np.linalg.norm(left_hand_pos - obj_pos)
        dist_right = np.linalg.norm(right_hand_pos - obj_pos)
        old_distance = 0
        use_right_hand = dist_right < dist_left
        gripper_link_indices = []
        if use_right_hand:
            arm_indices = [32, 33, 34, 35, 36, 37, 38]
            pickup_joints = [38]
            approach_joints = [32, 33, 34, 35, 36, 37]
            gripper_link_indices = [env.joint_index["r_moving_tip"], env.joint_index["r_fixed_tip"]]
          #  gripper_joint_indices = pickup_joints[-2:]  # last two gripper joints [37, 38]
        else:
            arm_indices = [22, 23, 24, 25, 26, 27, 28]
            pickup_joints = [28]
            approach_joints = [22, 23, 24, 25, 26 ,27]
            gripper_link_indices = [env.joint_index["l_moving_tip"], env.joint_index["l_fixed_tip"]]
           # gripper_joint_indices = pickup_joints[-2:]
        pos1 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
        pos2 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])
        pb.setCollisionFilterPair(env.robot_id, env.robot_id, gripper_link_indices[0], gripper_link_indices[1], enableCollision=True)
        joint_index_map = {joint_idx: i for i, joint_idx in enumerate(arm_indices)}
        agent.model.pickup_joints_idx = [joint_index_map[j] for j in pickup_joints]
        agent.model.approach_joints_idx = [joint_index_map[j] for j in approach_joints]
        joint_limits = get_joint_limits(env.robot_id, arm_indices)
        # Compute midpoint

        #voxels, corner = learner.object_to_voxels(obj)
        v_f = world_part_positions = get_object_part_world_positions(obj)
        midpoint = (pos1 + pos2) / 2.0
        v_f_np = np.array(v_f)
        v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)  # shape (n_voxels, 3)
        distances = torch.norm(v_f_tensor - midpoint, dim=1)
        old_distance = torch.min(distances)
        state = make_state(state_angles, obj_pos, obj_orientation,use_right_hand,v_f,pos1,pos2).to(device)
        #print(len(state))
        done = False

        states, actions, log_probs, rewards_list, values, next_values,dis, dones = [], [], [], [], [], [], [],[]
        steps = 0
        arewardlist = []
        prewardlist = []
        total_approach_reward = []
        total_pickup_reward = []
        device = next(agent.model.parameters()).device

        while steps<=50:

            state_tensor = state.float()
            action, log_prob ,std_dev,raw_action = agent.model.get_action(state_tensor)
            next_angles = env.get_position()
            if use_right_hand:
                arm_indices = [32, 33, 34, 35, 36, 37, 38]
                pickup_joints = [38]
                approach_joints = [32, 33, 34, 35,36,37]
                gripper_link_indices = [env.joint_index["r_moving_tip"], env.joint_index["r_fixed_tip"]]

            else:
                arm_indices = [22, 23, 24, 25, 26, 27, 28]
                pickup_joints = [28]
                approach_joints = [22, 23, 24, 25, 26,27]
                gripper_link_indices = [env.joint_index["l_moving_tip"], env.joint_index["l_fixed_tip"]]


            old_obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
            action_unnorm = unnormalize_actions(action, arm_indices, joint_limits)
            for i, idx in enumerate(arm_indices):
                next_angles[idx] = action_unnorm[i]
                #next_angles[idx]= action[i]

            env.goto_position(list(next_angles))

            next_state_angles = env.get_position()
            next_obj_pos,next_obj_orientation = pb.getBasePositionAndOrientation(obj_id)

            v_f = world_part_positions = get_object_part_world_positions(obj)
            pos1 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[0])[0])
            pos2 = tr.tensor(pb.getLinkState(env.robot_id, gripper_link_indices[1])[0])
            next_state = make_state(next_state_angles, next_obj_pos, next_obj_orientation,use_right_hand,v_f,pos1,pos2).to(device)
            areward, preward, graspcheck, new_distance, total_reward = rewards_potential(env,obj_pos,obj_id,gripper_link_indices,v_f,old_distance,scale_factor=22)
            old_distance = new_distance
            dis.append(new_distance)
            reward = areward+preward

            total_approach_reward.append(torch.tensor(areward.item()))
            total_pickup_reward.append(torch.tensor(preward.item()))

            reward = reward - 0.001*steps
            done = graspcheck
            if steps>50:
                done=True
            steps=steps+1
            if graspcheck == True and steps<=50:
                reward+= (50-steps)*0.01
            states.append(state_tensor)
            actions.append(raw_action)
            log_probs.append(log_prob)
            rewards_list.append(reward)
            arewardlist.append(areward)
            prewardlist.append(preward)
            values.append(agent.model(state_tensor)[2].detach())
            dones.append(torch.tensor(done, dtype=torch.float32))
            state = next_state

        with torch.no_grad():
            final_value = agent.model(state.float())[2].squeeze()
        values = torch.stack(values).squeeze().to(device)
        next_values = torch.cat([values[1:], final_value.unsqueeze(0)], dim=0).to(device)
        rewards_list = tr.tensor(rewards_list, dtype=tr.float32, device=device)
        arewardlist = tr.tensor([r.item() if isinstance(r, tr.Tensor) else r for r in arewardlist], dtype=tr.float32,
                                device=device)
        prewardlist = tr.tensor([r.item() if isinstance(r, tr.Tensor) else r for r in prewardlist], dtype=tr.float32,
                                device=device)
        dis = tr.tensor(dis, dtype=tr.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Ensure values, next_values, dones are on device:
        values = values.to(device)
        next_values = next_values.to(device)
        dones = dones.to(device)
        agent.update_avg_rewards(np.asarray(total_approach_reward), np.asarray(total_pickup_reward))
        #agent.last_pickup_reward_avg = total_pickup_reward/50
        #agent.last_approach_reward_avg = total_approach_reward/50
        agent.update(tr.stack(states).to(device), tr.stack(actions), tr.stack(log_probs).to(device),rewards_list,arewardlist,prewardlist,dis, values, next_values, dones)
        #suppress_cpp_output_stop()
        if episode%10 == 0:
            print(f"Episode {episode}, Tot_reward: {sum(rewards_list)}, ApproachReward: {sum(arewardlist)/len(arewardlist)}, Pickreward: {sum(prewardlist)/len(prewardlist)}")
        if episode % 20 == 0:
            print(f"Episode {episode}, StdDev: {std_dev[0]}")
        rewards_out.append(sum(rewards_list))
        log_p.append(log_probs)
        std_d.append(std_dev)
        areward_out.append(sum(arewardlist))
        preward_out.append(sum(prewardlist))
        if episode % 1000 == 1 and episode!=1:
            append_rewards_to_pickle(log_p, PICKLE_FILE2)
            #print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            log_p.clear()

        if episode % 1000 == 1 and episode!=1:
            append_rewards_to_pickle(std_d, PICKLE_FILE3)
            #print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            std_d.clear()

        if episode % 1000 == 1 and episode!=1:
            append_rewards_to_pickle(rewards_out, PICKLE_FILE)
            print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            rewards_out.clear()
        if episode % 1000 == 1 and episode!=1:
            append_rewards_to_pickle(prewardlist, PICKLE_FILE5)
          #  print(f"Appended {len(prewardlist)} rewards at iteration {episode}")
            preward_out.clear()
        if episode % 1000 == 1 and episode!=1:
            append_rewards_to_pickle(arewardlist, PICKLE_FILE4)
            #print(f"Appended {len(arewardlist)} rewards at iteration {episode}")
            areward_out.clear()
        env.close()
        if episode % 10000 == 0 and episode != 0:
            torch.save({
                'epoch': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'object': obj.positions
            }, 'ppo_checkpoint_2.pth')
        episode+=1
