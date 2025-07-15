import os
import sys
import numpy as np
import pybullet as pb
import torch
import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch import distributions as dist

sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
from ergo import PoppyErgoEnv
#import tabletop as tt
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
import BaselineLearner
from operator import add
import platform


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        #self.sig = nn.functional.sigmoid()
        self.std = nn.Parameter(tr.ones(action_dim)) # Trainable standard deviation

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor(features)
        value = self.critic(features)
        std = torch.sigmoid(self.std)
        return mean, std, value
w
    def get_action(self, state):
        mean, std, _ = self.forward(state)
       # mean = torch.tanh(mean)
        dist = tr.distributions.Normal(mean, std)
        action = dist.rsample()
        action_norm = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action_norm, log_prob,action

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-6, gamma=0.99, clip_eps=0.2, epochs=10):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs

    def compute_advantage(self, rewards, values, next_values, dones):
        advantages, returns = [], []
        advantage = 0
        for i in reversed(range(len(rewards))):
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error + self.gamma * 0.95 * advantage
            returns.insert(0, advantage + values[i])
            advantages.insert(0, advantage)
        return tr.tensor(advantages), tr.tensor(returns)

    def update(self, states, actions, old_log_probs, rewards, values, next_values, dones):
        advantages, returns = self.compute_advantage(rewards, values, next_values, dones)
        advantages = advantages.detach()
        returns = returns.detach()
        old_log_probs = old_log_probs.detach()

        for _ in range(self.epochs):
            mean, std, new_values = self.model(states)
            dist = tr.distributions.Normal(mean, std)

            # Unsquash actions using atanh (inverse tanh)
           # eps = 1e-6
          #  raw_actions = torch.atanh(actions.clamp(min=-1 + eps, max=1 - eps))

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            # Apply log prob correction
            new_log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
def make_state(angles, object_pos, obj_orientation,use_right_hand):
    objp = tr.tensor(object_pos)
    objo = tr.tensor(obj_orientation)
    rangles = tr.tensor(angles)
    use_right_hand_value = int(use_right_hand)  # converts True → 1, False → 0
    use_right_hand_tensor = tr.tensor([use_right_hand_value], dtype=tr.float32)
    return tr.cat((rangles, objp, objo,use_right_hand_tensor))

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
    if current_obj_pos[2] > 0.3:  # adjust threshold based on object/table height
        reward += 10.0  # bonus for picking up the object
    return reward
def rewards(env, obj_pos, obj_id, use_right_hand):
    # Calculate approach reward
    max_dist=1.0
    if use_right_hand:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
    else:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])

    gripper_midpoint = (tip1 + tip2) / 2.0
    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]

    return total_reward


def get_joint_limits(robot_id, joint_indices):

    joint_limits = {}
    for idx in joint_indices:
        info = pb.getJointInfo(robot_id, idx)
        joint_limits[idx] = (info[8], info[9])  # (lower_limit, upper_limit)
    return joint_limits

def unnormalize_actions(normalized_action, joint_indices, joint_limits):

    unnormalized = []
    for i, idx in enumerate(joint_indices):
        low, high = joint_limits[idx]
        # Scale from [-1, 1] → [low, high]
        scaled = (normalized_action[i] + 1.0) * 0.5 * (high - low) + low
        unnormalized.append(scaled)
    return unnormalized


import MultObjPick
import pybullet as pb
if __name__ == "__main__":
   # env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=False)
    state_dim = 50  # Adjusted for robot state representation
    action_dim = 7  # Assuming 7 joints as actions
    agent = PPOAgent(state_dim, action_dim)
    rewards_out = []
   # print(f"Joint {i}: {info[1].decode('utf-8')}")
    for episode in range(1000000):


        voxel_size = 0.015  # dimension of each voxel
        num_prll_prcs = 5
        dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
        n_parts = 8
        rgb = [(.75, .25, .25)] * n_parts
        gen0_results = []
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        #suppress_cpp_output_start()

        grasp_width = 1  # distance between grippers in voxel units
        # voxel_size = 0.015  # dimension of each voxel
        table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface
        learner = BaselineLearner.BaselineLearner(grasp_width, voxel_size)
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
        voxels, voxel_corner = learner.object_to_voxels(obj)
        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)
        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)
        state_angles = env.get_position()
        for i in range(pb.getNumJoints(env.robot_id)):
            info = pb.getJointInfo(env.robot_id, i)
            print(f"Joint {i}: {info[1].decode('utf-8')}")
       # obj_pos = pb.getBasePositionAndOrientation(tt.add_table())[0]
       # obj_orientation = pb.getBasePositionAndOrientation(tt.add_table())[1]
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        obj_pos, obj_orientation = pb.getBasePositionAndOrientation(obj_id)
        # pos of both hands
        left_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        right_hand_pos = np.array(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])

        # check distance and choose hand
        dist_left = np.linalg.norm(left_hand_pos - obj_pos)
        dist_right = np.linalg.norm(right_hand_pos - obj_pos)
        use_right_hand = dist_right < dist_left

        state = make_state(state_angles, obj_pos, obj_orientation,use_right_hand)
        done = False

        states, actions, log_probs, rewards_list, values, next_values, dones = [], [], [], [], [], [], []
        steps = 0




        while not done:
            if steps>20:
                done=True
            steps=steps+1
            state_tensor = state.float()
            action, log_prob,raw_act = agent.model.get_action(state_tensor)
            next_angles = env.get_position()
            if use_right_hand:
                arm_indices = [32, 33, 34, 35, 36, 37, 38]
            else:
                arm_indices = [22, 23, 24, 25, 26, 27, 28]

            joint_limits = get_joint_limits(env.robot_id, arm_indices)
            action_unnorm = unnormalize_actions(action, arm_indices, joint_limits)
            for i, idx in enumerate(arm_indices):
                next_angles[idx] = action_unnorm[i]
                #next_angles[idx]= action[i]

            env.goto_position(list(next_angles))

            next_state_angles = env.get_position()
            next_obj_pos,next_obj_orientation = pb.getBasePositionAndOrientation(obj_id)
            next_state = make_state(next_state_angles, next_obj_pos, next_obj_orientation,use_right_hand)

            reward = rewards(env, obj_pos,obj_id,use_right_hand)
            reward = reward - steps

            states.append(state_tensor)
            actions.append(raw_act)
            log_probs.append(log_prob)
            rewards_list.append(reward)
            values.append(agent.model(state_tensor)[2].detach())
            dones.append(torch.tensor(done, dtype=torch.float32))
            state = next_state

        with torch.no_grad():
            final_value = agent.model(state.float())[2].squeeze()
        values = torch.stack(values).squeeze()
        next_values = torch.cat([values[1:], final_value.unsqueeze(0)], dim=0)

        agent.update(tr.stack(states), tr.stack(actions), tr.stack(log_probs),
                     rewards_list, values, next_values, dones)
        #suppress_cpp_output_stop()

        print(f"Episode {episode}, Reward: {sum(rewards_list)}")
        rewards_out.append(sum(rewards_list))
        env.close()
        if episode % 50000 == 0 and episode != 0:
            torch.save(agent.model.state_dict(),f"ppo_model_ep_{episode}.pth")
