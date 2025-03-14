#import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys,os
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
import BaselineLearner
from operator import add

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(49,256)
        self.lin2 = nn.Linear(256,1024)
        self.fc = nn.Linear(1024,168) #reshape 168 into 4x42
        self.fc_to_action_space = nn.Linear(168,28)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        l1 = self.lin1(x)
        l2 = self.lin2(self.relu(l1))
        out = self.fc(self.relu(l2))
        out1 = self.relu(self.fc_to_action_space(out))
        out2 = self.sig(out1) # -1 to +1

##### sometehing missing here pds
        return out2

BATCH_SIZE = 2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
env= []
# Get number of actions from gym action space
n_actions = 8
# Get the number of state observations
#state, info = env.reset()
#n_observations = len(state)
policy_net = Model_1().to(device)
#policy_net_hands = Model_Hands().to(device)
target_net = Model_1().to(device)
#target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
#optimizer_hands = optim.AdamW(policy_net_hands.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
action_space = ["xplus", "xminus", "yplus", "yminus", "zplus", "zminus", "open", "close"]
import random

def select_action(state,action,high,low):
    global steps_done
    mu, sigma = 0, 0.2  # mean and standard deviation
    s = [np.random.normal(mu, sigma) for i in range(28)]
    sample = random.random()
    high = np.asarray(high)
    low = np.asarray(low)
    random_change = ((2*(s-low))/(high-low))-1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    eps_threshold = 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)
    else:
        return random_change





def processaction(action,ft_pos,mt_pos,units):
    if action == 0:
        ft_pos[0]=ft_pos[0] - units
        mt_pos[0] = mt_pos[0] - units
    elif action == 1:
        ft_pos[0]=ft_pos[0] + units
        mt_pos[0] = mt_pos[0] + units
    elif action == 2:
        ft_pos[1] = ft_pos[1] - units
        mt_pos[1] = mt_pos[1] - units
    elif action == 3:
        ft_pos[1] = ft_pos[1] + units
        mt_pos[1] = mt_pos[1] + units
    elif action == 4:
        ft_pos[2] = ft_pos[2] - units
        mt_pos[2] = mt_pos[2] - units
    elif action == 5:
        ft_pos[2] = ft_pos[2] + units
        mt_pos[2] = mt_pos[2] + units
    else:
        return ft_pos,mt_pos
    return ft_pos,mt_pos

def process_hands(action):
    if action == 0:
        return "lr"
    else:
        return "rl"
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def rewards(p1,p2):
    p1_i = np.asarray(p1)
    p2_i = np.asarray(p2)
    squared_dist = np.sum((p1_i - p2_i) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    rewards = -dist
    return rewards

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.reshape(BATCH_SIZE,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def optimize_model_hands():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    #optimizer_hands.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #optimizer_hands.step()


check_UL = []
check_LL = []
if __name__ == "__main__":
    num_episodes = 100
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        import MultObjPick
        import pybullet as pb

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
        check = env.joint_name

        for i in range(env.num_joints):
            check_UL.append(pb.getJointInfo(env.robot_id,i)[8])
            check_LL.append(pb.getJointInfo(env.robot_id,i)[9])
        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height:
            pb.removeBody(obj_id)
            continue
        M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
        M = np.array(M).reshape(3, 3)
        rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
        grip_point = rest_coords[0]
        t,arm =  learner.get_pick_trajectory_variation(exp.env, grip_point)
        for i in range(len(t)-3):
            #trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            exp.env.goto_position(t[i], duration=2)
            exp.env.goto_position(t[i], duration=.1)  # in case it needs a little more time to converge
            #trajectory = learner.get_pick_trajectory(exp.env, grip_points)



        #state is combination of env variables.
        start_angles = env.get_position()
        state = np.concatenate((start_angles,rest_pos,rest_orn), axis=None)
        #goto position close to obj ( center of voxel)
       # state.append(start_angles,rest_pos,rest_orn)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        links = [
            env.joint_index[f"{arm}_moving_tip"],
            env.joint_index[f"{arm}_fixed_tip"],
           # env.joint_index[f"{arm}_fixed_knuckle"], # for top-down approach
        ]

        # joints that are free for IK
        free = list(range(env.num_joints))
        # links with targets are not considered free
        for idx in links: free.remove(idx)
        # keep reasonable posture
        free.remove(env.joint_index["head_z"]) # neck
        free.remove(0) # waist



   #     state = torch.from_numpy(state)
        timer = 0
        for t in count():
            timer= timer +1
            lft_pos = list(pb.getJointState(env.robot_id, env.joint_index["l_fixed_tip"]))
            lmt_pos = list(pb.getJointState(env.robot_id, env.joint_index["l_moving_tip"]))
            rft_pos = list(pb.getJointState(env.robot_id, env.joint_index["r_fixed_tip"]))
            rmt_pos = list(pb.getJointState(env.robot_id, env.joint_index["r_moving_tip"]))
            lft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
            lmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
            rft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
            rmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
            rh_pos = torch.mul(torch.tensor(list(map(add, rft_pos, rmt_pos))), 0.5)
            lh_pos = torch.mul(torch.tensor(list(map(add, lft_pos, lmt_pos))), 0.5)


            if arm == "l":
                action_select = select_action(state, action_space,check_UL[14:],check_LL[14:])
               # action_process = processaction(action_select,lft_pos,lmt_pos,0.01)
            else:
                action_select = select_action(state, action_space, check_UL[14:], check_LL[14:])
                #action_process = processaction(action_select,rft_pos,rmt_pos,0.01)

            #npa = np.asarray(action_process, dtype=np.float32)
            #traj = learner.get_pick_trajectory(exp.env,npa)
            #action = traj[3]

            targets = action_select     # + list(pb.getJointStates(env.robot_id,[i for i in range(14:env.num_joints)]))
            all_joints = [i for i in range(0,42)]
            cur_angles = list(pb.getJointStates(env.robot_id,all_joints))
            c_ang = [cur_angles[i][0] for i in range(len(cur_angles))]
            targets_new = c_ang.copy()
            for k in range(14,len(cur_angles)):
                if targets[i]==0:
                    continue
                else:
                    targets_new[i] = np.asarray(targets)[i-14] + np.asarray(c_ang)[i]


           # targets[2, 2] += .05
            #all_targets = env.forward_kinematics()
           # all_targets[links] = targets
            #all_links = np.array([j for j in range(env.num_joints) if j not in free])
          #  all_targets = all_targets[all_links]
            #angles = env.inverse_kinematics(all_links, all_targets, num_iters=1000)
            angles = np.asarray(targets_new)
            env.step(angles)
            action = torch.from_numpy(angles)
            new_angles = env.get_position()
            o_pos, o_orn = pb.getBasePositionAndOrientation(obj_id)

            if arm == "l":
                lft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
                lmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
                rft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
                rmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
                rh_pos = torch.mul(torch.tensor(list(map(add, rft_pos, rmt_pos))), 0.5)
                lh_pos = torch.mul(torch.tensor(list(map(add, lft_pos, lmt_pos))), 0.5)
                reward = rewards(o_pos, lh_pos)
            else:
                lft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
                lmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
                rft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
                rmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
                rh_pos = torch.mul(torch.tensor(list(map(add, rft_pos, rmt_pos))), 0.5)
                lh_pos = torch.mul(torch.tensor(list(map(add, lft_pos, lmt_pos))), 0.5)
                reward = rewards(o_pos, rh_pos)
            #done = terminated or truncated
            observation = np.concatenate((new_angles, o_pos, o_orn), axis=None)
            #if terminated:
             #   next_state = None
            #else:

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, torch.tensor(angles), next_state, torch.tensor(reward))
            print(action_select,reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            #Model_1.state_dict()
           # optimizer.state_dict()
           # optimize_model_hands()
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if timer>200:
               # episode_durations.append(t + 1)
               # plot_durations()
                break
        env.close()
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()