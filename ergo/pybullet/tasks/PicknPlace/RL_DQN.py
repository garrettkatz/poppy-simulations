import gymnasium as gym
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

        self.lin1 = nn.Linear(78,256)
        self.lin2 = nn.Linear(256,1024)
        self.fc = nn.Linear(1024,168) #reshape 168 into 4x42
        self.fc_to_action_space = nn.Linear(168,8)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        l1 = self.lin1(x)
        l2 = self.lin2(self.relu(l1))
        out = self.fc(self.relu(l2))
        out1 = self.relu(self.fc_to_action_space(out))
        out2 = self.relu(self.sig(out1))
##### sometehing missing here pds
        return out2

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(36, 128)  # Relative pos of all voxels of object
        self.lin2 = nn.Linear(36, 128)    # Absolute position of all voxels of object
        self.lin3 = nn.Linear(6,100)   # Grip points flatten , xyz for 2 fingers
        self.fc = nn.Linear(356, 168)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, ip_angles,ip_graspVoxel,ip_Object):
        l1 = self.lin1(ip_angles.float())
        l2 = self.lin2(ip_graspVoxel.float())
        l3 = self.lin3(ip_Object.float())
        l4 = np.concat(l1,l2,l3)
        out = self.fc(l4)
        return out

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
env= []
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
#n_observations = len(state)
policy_net = Model_1().to(device)
target_net = Model_1().to(device)
#target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long




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
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    rewards = 1/dist
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
    reward_batch = torch.cat(batch.reward)

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
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
import BaselineLearner

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
        voxel_size = 0.015  # dimension of each voxel
        table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface
        learner = BaselineLearner(grasp_width, voxel_size)
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
        obj_id = exp.Spawn_Object(obj)
        # Mutant = obj.MutateObject()
        voxels, voxel_corner = learner.object_to_voxels(obj)
        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)
        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)
        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height:
            pb.removeBody(obj_id)
            exp.env.close()

        action_space = ["Handselection""xplus", "xminus", "yplus", "yminus", "zplus", "zminus", "open", "close"]
        #state is combination of env variables.

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = rewards(state,observation)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
               # episode_durations.append(t + 1)
               # plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()