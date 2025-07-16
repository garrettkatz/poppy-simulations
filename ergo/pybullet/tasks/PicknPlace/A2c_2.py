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
        self.std = nn.Parameter(tr.zeros(action_dim))  # Trainable standard deviation

        self.approach_joints_idx = []
        self.pickup_joints_idx = []

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor(features)
        value = self.critic(features)
        std = torch.sigmoid(self.std) * 0.3
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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.ema_alpha = 0.5
        self.last_approach_reward_avg =0
        self.last_pickup_reward_avg = 0
        #self.approach_joints_idx = []
        #self.pickup_joints_idx = []
        self.ema = 0.5
    def compute_advantage(self, rewards, values, next_values, dones):
        advantages, returns = [], []
        advantage = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * 0.95 * advantage
            returns.insert(0, advantage + values[i])
            advantages.insert(0, advantage)
        return tr.tensor(advantages), tr.tensor(returns)

    def update(self, states, actions, old_log_probs, rewards, approach_rewards, pickup_rewards, values, next_values, dones):
        advantages, returns = self.compute_advantage(rewards, values, next_values, dones)

        # FIX: Detach to avoid backward-through-graph error
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        advantages = advantages.detach()
        returns = returns.detach()
        old_log_probs = old_log_probs.detach()
        states = states.detach()
        actions = actions.detach()
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
            normalized_approach = torch.tensor(approach_rewards).sum(dim=-1).mean()
            normalized_approach = torch.clamp(torch.tensor(self.last_approach_reward_avg), 0.0, 1.0) #assert better
            normalized_pickup = torch.clamp(torch.tensor(self.last_pickup_reward_avg) / 10.0, 0.0, 1.0)
            inv_approach = 1.0 - normalized_approach
            inv_pickup = 1.0 - normalized_pickup
            min_beta = 0.001
            max_beta = 0.01
            if normalized_approach>0.7:
                current_approach_entropy_coeff = min_beta + (max_beta - min_beta) * inv_approach
            else:
                current_approach_entropy_coeff =max_beta

            if normalized_pickup>0.3:
                current_pickup_entropy_coeff = min_beta + (max_beta - min_beta) * inv_pickup
            else:
                current_pickup_entropy_coeff = max_beta

            entropy_approach = joint_entropies[:, self.model.approach_joints_idx].sum(dim=-1).mean()
            entropy_pickup = joint_entropies[:, self.model.pickup_joints_idx].sum(dim=-1).mean()

            total_entropy_bonus = current_approach_entropy_coeff * entropy_approach + current_pickup_entropy_coeff * entropy_pickup

            policy_loss = policy_loss - total_entropy_bonus

            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

    def update_avg_rewards(self, current_approach_reward, current_pickup_reward):
        self.last_approach_reward_avg = (
                (1 - self.ema_alpha) * self.last_approach_reward_avg + self.ema_alpha * current_approach_reward
        )
        self.last_pickup_reward_avg = (
                (1 - self.ema_alpha) * self.last_pickup_reward_avg + self.ema_alpha * current_pickup_reward
        )

def make_state(angles, object_pos, obj_orientation,use_right_hand,obj_part_positions=None):
    objp = tr.tensor(object_pos)
    objo = tr.tensor(obj_orientation)
    rangles = tr.tensor(angles)
    use_right_hand_value = int(use_right_hand)  # converts True → 1, False → 0
    use_right_hand_tensor = tr.tensor([use_right_hand_value], dtype=tr.float32)
    # Base state (15D): joint angles + object position + orientation + hand indicator
    base_state = tr.cat((rangles, objp, objo, use_right_hand_tensor))

    # Optionally concatenate voxel-based object features
    if obj_part_positions is not None:
        part_tensor = torch.tensor(np.array(obj_part_positions), dtype=tr.float32).flatten()
        return torch.cat((part_tensor,base_state))
    else:
        return base_state


def rewards1(env, objpos):
    rh_pos = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
    distance = tr.norm(rh_pos - tr.tensor(objpos[0]))
    return -distance  # Negative distance as reward

def rewards2(env, objpos, obj_id, use_right_hand=True):
    # Get current hand position
    if use_right_hand:
        hand_pos = torch.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
    else:
        hand_pos = torch.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
    obj_pos = torch.tensor(objpos[0])
    distance = torch.norm(hand_pos - obj_pos)
    # Inverse distance reward
    reward = 1.0 / (distance + 1e-4)
    # Pick-up bonus: if object has moved significantly in Z
    current_obj_pos = torch.tensor(pb.getBasePositionAndOrientation(obj_id)[0])
    if current_obj_pos[2] > 0.1:  # adjust threshold based on object/table height
        reward += 10.0  # bonus for picking up the object
    return reward

def rewards(env, obj_pos, obj_id, use_right_hand):
    # Calculate approach reward
    max_dist=1.0
    g_check = False
    gripper_link_indices = []
    if use_right_hand:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
        gripper_link_indices = [env.joint_index["r_moving_tip"], env.joint_index["r_fixed_tip"]]
    else:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        gripper_link_indices = [env.joint_index["l_moving_tip"], env.joint_index["l_fixed_tip"]]
    tip_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id,
                                       linkIndexA=gripper_link_indices[0], linkIndexB=gripper_link_indices[1])

    tip_distance = torch.norm(tip1 - tip2)
    distance_diff = torch.abs(tip_distance - 0.015)
    gripper_midpoint = (tip1 + tip2) / 2.0
    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
    distance = torch.norm(gripper_midpoint - torch.tensor(pb.getBasePositionAndOrientation(obj_id)[0]))
    approach_reward = 1.0 / (distance*10 + 1e-4)
    approach_reward = torch.clamp(approach_reward, max=10.0)
    approach_reward =approach_reward/10 #normalize
    voxel_distance_bonus = max(0.0, 0.1 * (1 - distance_diff / 0.015))

    # Check if object is picked up (height threshold)
    pickup_reward = (10) if obj_height > obj_pos[2] + 0.2 else (obj_height - obj_pos[2])*10 # Adjust threshold as needed
    pickup_reward = torch.tensor(pickup_reward)
    pickup_reward += voxel_distance_bonus
    pickup_reward = torch.clamp(pickup_reward, min=0.0)
    if len(tip_contacts)>0:
        pickup_reward-=1

        #print("penalty for overlapping grippers")
    for link_idx in gripper_link_indices: # This loop now only checks the specified gripper tips
        contact_points = pb.getContactPoints(env.robot_id, obj_id, link_idx, -1)
        if len(contact_points) ==1 :
            print("contact at 1 points,approach reward at poc: ",approach_reward)
            pickup_reward+=1
        if len(contact_points) > 1:
            print("contact at 2 points,approach reward at poc : ",approach_reward)
            pickup_reward += 2
            #break

    #table_height = table_position()[2] + table_half_extents()[2]
   # pickup_reward = (10 +(obj_height - obj_pos[2])*10) if obj_height > obj_pos[2] + 0.2 else (obj_height - obj_pos[2])*10 # Adjust threshold as needed
    if obj_height > obj_pos[2] + 0.2:
        g_check = True
        print("Pick success")
    total_reward = approach_reward + pickup_reward
    total_reward = torch.clamp(total_reward,min=0.0)
    return (approach_reward),(pickup_reward),g_check


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

PICKLE_FILE = 'rewards9thexperiment_contactP_entropy_rewards.pickle'
PICKLE_FILE2 = 'rewards9thexperiment_contactP_entropy_logprob.pickle'
PICKLE_FILE3 = 'rewards9thdexperiment_contactP_entropy_stddev.pickle'

PICKLE_FILE4 = 'rewards9thexperiment_contactP_entropy_arewards.pickle'

PICKLE_FILE5 = 'rewards9thexperiment_contactP_entropy_prewards.pickle'
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
   # env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=False)
    state_dim = 80  # Adjusted for robot state representation
    action_dim = 7  # Assuming 7 joints as actions
    agent = PPOAgent(state_dim, action_dim)
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
    #learner = BaselineLearner.BaselineLearner(grasp_width, voxel_size)
    obj.GenerateObject(dims, n_parts, [0, 0, 0])
    #voxels, voxel_corner = learner.object_to_voxels(obj)
    # get candidate grasp points
    #cands = learner.collect_grasp_candidates(voxels)
    # convert back to simulator units
    #coords = learner.voxel_to_sim_coords(cands, voxel_corner)

    if resume== 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load('ppo_model_entropy_ep_20000.pth', map_location=device, weights_only=True)
        #print(checkpoint.keys())
        agent.model.load_state_dict(state_dict)
        #agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = 0
   # print(f"Joint {i}: {info[1].decode('utf-8')}")
    while episode < max_episodes:

        #print("Episode:",episode)

        #suppress_cpp_output_start()


        # voxel_size = 0.015  # dimension of each voxel
        table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

        Num_success = 0
        Num_Grips_attempted = 0
        Result = []
        num_objects = 1

        exp = MultObjPick.experiment()
        exp.CreateScene()
        env = exp.env

        pb.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=-1.2,
            cameraPitch=-39.0,
            cameraTargetPosition=(0., 0., 0.),
        )

        #  scaling = ((2*(x-low))/(high-low))-1
        obj_id = exp.Spawn_Object(obj)
        # Mutant = obj.MutateObject()

        state_angles = env.get_position()
      #  for i in range(pb.getNumJoints(env.robot_id)):
          #  info = pb.getJointInfo(env.robot_id, i)
          #  print(f"Joint {i}: {info[1].decode('utf-8')}")
       # obj_pos = pb.getBasePositionAndOrientation(tt.add_table())[0]
       # obj_orientation = pb.getBasePositionAndOrientation(tt.add_table())[1]
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        if episode !=0:

            exp.env.settle(exp.env.get_position(), seconds=0.2)
        obj_pos, obj_orientation = pb.getBasePositionAndOrientation(obj_id)
        #check body falling before start of episode
        if obj_pos[2] < table_height:
            pb.removeBody(obj_id)
            print("Obj fell off on episode,restarting",episode)
            #episode = episode-1
            obj = MultObjPick.Obj(dims, n_parts, rgb)
            env.close()
            continue

        # pos of both hands
        left_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        right_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])

        # check distance and choose hand
        dist_left = np.linalg.norm(left_hand_pos - obj_pos)
        dist_right = np.linalg.norm(right_hand_pos - obj_pos)
        use_right_hand = dist_right < dist_left
        #voxels, corner = learner.object_to_voxels(obj)
        v_f = world_part_positions = get_object_part_world_positions(obj)

        state = make_state(state_angles, obj_pos, obj_orientation,use_right_hand,v_f)
        #print(len(state))
        done = False

        states, actions, log_probs, rewards_list, values, next_values, dones = [], [], [], [], [], [], []
        steps = 0
        arewardlist = []
        prewardlist = []
        total_approach_reward = 0.0
        total_pickup_reward = 0.0



        while not done:

            state_tensor = state.float()
            action, log_prob ,std_dev,raw_action = agent.model.get_action(state_tensor)
            next_angles = env.get_position()
            if use_right_hand:
                arm_indices = [32, 33, 34, 35, 36, 37, 38]
                pickup_joints = [36, 37, 38]
                approach_joints = [32, 33, 34, 35,36]
            else:
                arm_indices = [22, 23, 24, 25, 26, 27, 28]
                pickup_joints = [26, 27, 28]
                approach_joints = [22, 23, 24, 25, 26]

            joint_index_map = {joint_idx: i for i, joint_idx in enumerate(arm_indices)}
            agent.model.pickup_joints_idx = [joint_index_map[j] for j in pickup_joints]
            agent.model.approach_joints_idx = [joint_index_map[j] for j in approach_joints]
            joint_limits = get_joint_limits(env.robot_id, arm_indices)
            action_unnorm = unnormalize_actions(action, arm_indices, joint_limits)
            for i, idx in enumerate(arm_indices):
                next_angles[idx] = action_unnorm[i]
                #next_angles[idx]= action[i]

            env.goto_position(list(next_angles))

            next_state_angles = env.get_position()
            next_obj_pos,next_obj_orientation = pb.getBasePositionAndOrientation(obj_id)
            #next_obj_orientation = pb.getBasePositionAndOrientation(obj_id)
            v_f = world_part_positions = get_object_part_world_positions(obj)
            next_state = make_state(next_state_angles, next_obj_pos, next_obj_orientation,use_right_hand,v_f)

            areward,preward,graspcheck = rewards(env, obj_pos,obj_id,use_right_hand)
            #done = False  # Define termination condition
            #preward=preward if preward>0 else 0.0
            reward = areward+preward if preward > 0.7 else preward
            total_approach_reward += areward.item()
            total_pickup_reward += preward.item()
            reward = torch.clamp(reward, min=0.0)
            reward = reward - 0.01*steps
            done = graspcheck
            if steps>10:
                done=True
            steps=steps+1
            if graspcheck == True and steps<50:
                reward+= (50-steps)*0.1
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
        values = torch.stack(values).squeeze()
        next_values = torch.cat([values[1:], final_value.unsqueeze(0)], dim=0)
        agent.update_avg_rewards(total_approach_reward, total_pickup_reward)
        agent.last_pickup_reward_avg = total_pickup_reward/50
        agent.last_approach_reward_avg = total_approach_reward/50
        agent.update(tr.stack(states), tr.stack(actions), tr.stack(log_probs),rewards_list,arewardlist,prewardlist, values, next_values, dones)
        #suppress_cpp_output_stop()
        if episode%5 == 0:
            print(f"Episode {episode}, AvgReward: {sum(rewards_list)/len(rewards_list)}, ApproachReward: {sum(arewardlist)/len(arewardlist)}, Pickreward: {sum(prewardlist)/len(prewardlist)}")
        if episode % 10 == 0:
            print(f"Episode {episode}, StdDev: {std_dev[-1]}")
        rewards_out.append(sum(rewards_list))
        log_p.append(log_probs)
        std_d.append(std_dev)
        areward_out.append(sum(arewardlist))
        preward_out.append(sum(prewardlist))
        if episode % 100 == 1 and episode!=1:
            append_rewards_to_pickle(log_p, PICKLE_FILE2)
            #print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            log_p.clear()

        if episode % 100 == 1 and episode!=1:
            append_rewards_to_pickle(std_d, PICKLE_FILE3)
            #print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            std_d.clear()

        if episode % 100 == 1 and episode!=1:
            append_rewards_to_pickle(rewards_out, PICKLE_FILE)
            print(f"Appended {len(rewards_out)} rewards at iteration {episode}")
            rewards_out.clear()
        if episode % 100 == 1 and episode!=1:
            append_rewards_to_pickle(prewardlist, PICKLE_FILE5)
          #  print(f"Appended {len(prewardlist)} rewards at iteration {episode}")
            rewards_out.clear()
        if episode % 100 == 1 and episode!=1:
            append_rewards_to_pickle(arewardlist, PICKLE_FILE4)
            #print(f"Appended {len(arewardlist)} rewards at iteration {episode}")
            rewards_out.clear()
        env.close()
        if episode % 20000 == 0 and episode != 0:
            torch.save({
                'epoch': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, 'ppo_checkpoint.pth')
        episode+=1
