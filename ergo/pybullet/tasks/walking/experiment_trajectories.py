import sys, os
import pybullet as pb
import numpy as np
import pickle as pk

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

from phase_waypoints import get_waypoints, check_waypoints
from phase_trajectories import *

if __name__ == "__main__":

    num_samples = 30

    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)

    sample_data = []
    for sample in range(num_samples):
        print(f"sample {sample}...")

        # angle from vertical axis to flat leg in initial stance
        init_flat = .02*np.pi + np.random.normal(0, 0.5 * np.pi/180)
        # angle for abs_y joint in initial stance
        init_abs_y = np.pi/16 + np.random.normal(0, 0.5 * np.pi/180)
        # angle from swing leg to vertical axis in shift stance
        shift_swing = .05*np.pi + np.random.normal(0, 0.5 * np.pi/180)
        # angle of torso towards support leg in shift stance
        shift_torso = np.pi/5 + np.random.normal(0, 0.5 * np.pi/180)
        # angle from vertical axis to flat leg in push stance
        push_flat = -.00*np.pi + np.random.normal(0, 0.5 * np.pi/180)
        # angle from swing leg to vertical axis in push stance
        push_swing = -.10*np.pi + np.random.normal(0, 0.5 * np.pi/180)

        params = (init_flat, init_abs_y, shift_swing, shift_torso, push_flat, push_swing)
        waypoints = get_waypoints(env, *params)

        checks = check_waypoints(env, waypoints)

        trajectories = get_direct_trajectories(env, waypoints)
        num_points = [10, 10, 2, 2, 1]
        trajectories = constrained_interpolate(env, trajectories, num_points)
        trajectories = [make_arccos_durations(traj) for traj in trajectories]

        if not all(checks):
            phase_trajectory_figure(env, trajectories)

        trajectories = extend_mirrored_trajectory(env, trajectories)
        pypot_trajectories = tuple(map(env.get_pypot_trajectory, trajectories))
        with open(f'exp_normal/pypot_sample_trajectory_{sample}.pkl', "wb") as f: pk.dump(pypot_trajectories, f, protocol=2) 

        sample_data.append((params, checks, trajectories))

        print("  params:", params)
        print("  checks:", checks)

    env.close()

    with open(f'exp_normal/exp_traj_data.pkl', "wb") as f: pk.dump(sample_data, f) 


