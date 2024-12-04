import math
import random

import torch
import torch.nn as nn
import numpy as np
import sys,os
import pickle
import glob, os
import re

import matplotlib.pyplot as plt
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(78,256)
        self.lin2 = nn.Linear(256,1024)
        self.fc = nn.Linear(1024,168) #reshape 168 into 4x42
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        l1 = self.lin1(x)
        l2 = self.lin2(self.relu(l1))
        out = self.fc(self.relu(l2))
        out1 = self.sig(out)
        out2 = out1*(2*math.pi)
        out3 = out2.reshape(4,42)
        return out3

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

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class BaselineLearner:
    """
    grasp_width: max distance between grippers when grasping, in voxel units
    voxel_size: the width of each voxel, in simulator units
    """

    def __init__(self, grasp_width, voxel_size):
        self.grasp_width = grasp_width
        self.voxel_size = voxel_size

    """
    Get padded voxel representation of object
    """

    def object_to_voxels(self, obj):

        # get offsets from object base position to voxel centers
        voxel_offsets = np.array(obj.positions)

        # convert simulation units to grid (i,j,k) indices
        vijk = ((voxel_offsets - voxel_offsets.min(axis=0)) / self.voxel_size).round().astype(int)

        # pad grid to check that voxels next to contact points are empty
        pad = 1
        vijk += pad  # lower padding
        grid_size = vijk.max(axis=0) + self.grasp_width + pad + 1  # upper padding

        # fill in voxel grid
        voxels = np.zeros(tuple(grid_size))
        for (i, j, k) in vijk: voxels[i, j, k] = 1

        # also get coordinates for lower grid corner corner to convert back later
        voxel_corner = voxel_offsets.min(axis=0) - voxel_size / 2

        return voxels, voxel_corner

    """
    Input:
        voxels[i,j,k]: 1 if object includes voxel at (i,j,k), 0 otherwise
            assumes padding of 1, meaning boundary voxels are all 0
    Output:
        cands[n,i,:]: voxel grid coordinates of ith contact point in nth grasp, i in (0, 1)
    """

    def collect_grasp_candidates(self, voxels):
        candidates = []

        # try all grasp widths up to max
        for gw in range(1, self.grasp_width + 1):

            # pattern is like [0, 1, 0], 0 is where the fingertips go
            grasp_pattern = np.ones(gw + 2)
            grasp_pattern[[0, -1]] = 0

            for (i, j, k) in zip(*np.nonzero(voxels)):
                if (voxels[i - 1:i + gw + 1, j, k] == grasp_pattern).all():
                    candidates.append(((i, j + .5, k + .5), (i + gw, j + .5, k + .5)))
                if (voxels[i, j - 1:j + gw + 1, k] == grasp_pattern).all():
                    candidates.append(((i + .5, j, k + .5), (i + .5, j + gw, k + .5)))
                if (voxels[i, j, k - 1:k + gw + 1] == grasp_pattern).all():
                    candidates.append(((i + .5, j + .5, k), (i + .5, j + .5, k + gw)))

        return np.array(candidates)

    """
    Convert grip coordinates from voxel to simulation units
        candidates: as returned by collect_grasp_candidates
        voxel_corner: lower coordinates of voxel grid for conversion back to simulator units
    Output:
        coords[n,i,:]: simulator coordinates of ith contact point in nth grasp, i in (0, 1)
    """

    def voxel_to_sim_coords(self, candidates, voxel_corner):
        coords = (candidates - 1)  # undo padding
        coords = coords * self.voxel_size + voxel_corner  # rescale and recenter
        return coords

    """
    Create joint trajectory to pick up an object at specified grip_points
    input: grip_points[i]: xyz coordinates for ith finger
    output: trajectory[n]: joint angles for nth waypoint
    """

    def get_pick_trajectory(self, env, grip_points):

        # choose arm based on x-coordinate of grip points
        if grip_points.mean(axis=0)[0] > 0:
            arm, other_arm = "lr"
        else:
            arm, other_arm = "rl"

        # links with specified IK targets
        links = [
            env.joint_index[f"{arm}_moving_tip"],
            env.joint_index[f"{arm}_fixed_tip"],
            env.joint_index[f"{arm}_fixed_knuckle"],  # for top-down approach
        ]

        # joints that are free for IK
        free = list(range(env.num_joints))
        # links with targets are not considered free
        for idx in links: free.remove(idx)
        # keep reasonable posture
        free.remove(env.joint_index["head_z"])  # neck
        free.remove(0)  # waist

        # get waypoint targets
        grip_centers = grip_points.mean(axis=0, keepdims=True)
        open_down = grip_centers + 1.5 * (grip_points - grip_centers)
        closed_down = grip_centers + 0.9 * (grip_points - grip_centers)
        closed_up = closed_down + np.array([[0, 0, 0.1]])
        open_up = open_down + np.array([[0, 0, 0.1]])
        waypoints = (open_up, open_down, closed_down, closed_up)

        # try each finger at each contact point
        start_angles = env.get_position()
        max_error = [0] * 2
        trajectories = []
        for i, (fixed, moving) in enumerate(([0, 1], [1, 0])):

            # IK on each waypoint
            env.set_position(start_angles)
            angles = {-1: start_angles}
            for w in range(len(waypoints)):

                # targets for current assignment of fingers to contact points
                targets = waypoints[w][[moving, fixed, fixed], :]
                targets[2, 2] += .05  # xyz origin offset of fixed_knuckle in urdf

                # try to reach
                angles[w], out_of_range, error = env.partial_ik(links, targets, angles[w - 1], free, num_iters=3000)
                if angles[w][env.joint_index[f"{arm}_gripper"]] < 0.5:
                    continue
                max_error[i] = max(max_error[i], error)
                # max_error.append(max(max_error[i], error))

                # don't change other arm angles
                for joint_name in ("shoulder_x", "shoulder_y", "arm_z", "elbow_y"):
                    idx = env.joint_index[f"{other_arm}_{joint_name}"]
                    # angles[w][idx] = 0
                    angles[w][idx] = start_angles[idx]

                # env.set_position(angles[w])
                # print(max_error[i])
                # input('..')
            # angle[w][x] < 0 , continue
            trajectory = [angles[w] for w in range(len(waypoints))]
            # if trajectory[len(trajectory) - 1][env.joint_index[f"{arm}_gripper"]] > 0.5:
            #      continue
            # if trajectory[len(trajectory) - 1][env.joint_index[f"{other_arm}_gripper"]] > 0.5:
            #     continue
            trajectories.append(trajectory)
            env.set_position(start_angles)

        # choose trajectory with lower max error
        choice = np.argmin(max_error)

        trajectory = trajectories[choice]

        return trajectory

def rewards(p1,p2):
    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    rewards = 1/dist
    return rewards


def AttemptGripsRL(obj,gen):
    import MultObjPick
    import pybullet as pb

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
        return 0,-1
    # sys.exit(0)
    M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
    M = np.array(M).reshape(3, 3)
    rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
    #interm_result = []
    res = []
    num_grips_success = 0

    action_space = ["xplus","xminus","yplus","yminus","zplus","zminus","open","close"]
    # convert action space to coordinates

    #observation
    #select action based on eps
    steps_done = 0
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    sample = random.Random()
    if sample>eps_threshold:

