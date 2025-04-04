import os
import sys
import numpy as np
import pybullet as pb
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
        self.std = nn.Parameter(tr.ones(action_dim) * 0.1)  # Trainable standard deviation

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor(features)
        value = self.critic(features)
        std = self.std.exp()
        return mean, std, value

    def get_action(self, state):
        mean, std, _ = self.forward(state)
        dist = tr.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10):
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

        for _ in range(self.epochs):
            mean, std, new_values = self.model(states)
            dist = tr.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = tr.exp(new_log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = tr.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -tr.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def make_state(angles, object_pos, obj_orientation):
    objp = tr.tensor(object_pos[0])
    objo = tr.tensor(obj_orientation)
    rangles = tr.tensor(angles)
    return tr.cat((rangles, objp, objo))

def rewards(env, objpos):
    rh_pos = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
    distance = tr.norm(rh_pos - tr.tensor(objpos[0]))
    return -distance  # Negative distance as reward


import MultObjPick
import pybullet as pb
if __name__ == "__main__":
   # env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=False)
    state_dim = 48  # Adjusted for robot state representation
    action_dim = 9  # Assuming 9 joints as actions
    agent = PPOAgent(state_dim, action_dim)

    for episode in range(1000):


        voxel_size = 0.015  # dimension of each voxel
        num_prll_prcs = 5
        dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
        n_parts = 8
        rgb = [(.75, .25, .25)] * n_parts
        gen0_results = []
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])

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
       # obj_pos = pb.getBasePositionAndOrientation(tt.add_table())[0]
       # obj_orientation = pb.getBasePositionAndOrientation(tt.add_table())[1]

        state = make_state(state_angles, obj_pos, obj_orientation)
        done = False

        states, actions, log_probs, rewards_list, values, next_values, dones = [], [], [], [], [], [], []
        steps = 0
        while not done:
            state_tensor = state.float()
            action, log_prob = agent.model.get_action(state_tensor)
            next_angles = env.get_position()
            next_angles[27:36] = action
            env.goto_position(list(next_angles), 1)

            next_state_angles = env.get_position()
            next_obj_pos = pb.getBasePositionAndOrientation(tt.add_table())[0]
            next_obj_orientation = pb.getBasePositionAndOrientation(tt.add_table())[1]
            next_state = make_state(next_state_angles, next_obj_pos, next_obj_orientation)

            reward = rewards(env, next_obj_pos)
            done = False  # Define termination condition

            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards_list.append(reward)
            values.append(agent.model(state_tensor)[2].detach())

            state = next_state

        next_values.append(agent.model(state.float())[2].detach())

        agent.update(tr.stack(states), tr.stack(actions), tr.stack(log_probs),
                     rewards_list, values, next_values, dones)

        print(f"Episode {episode}, Reward: {sum(rewards_list)}")

        env.close()