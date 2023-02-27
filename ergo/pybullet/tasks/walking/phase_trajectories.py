import sys, os
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

from phase_waypoints import get_waypoints, phase_waypoint_figure, render_legs

# single goto_position command for each phase waypoint
#   returns trajectories = ( ..., (..., (duration, targets), ...), ...)
def get_direct_trajectories(env, waypoints):

    # move ankle before knee during swing to ensure ground clearance
    init, shift, push, kick = list(zip(*waypoints))[0]
    push_kick = push.copy()
    push_kick[env.joint_index['l_ankle_y']] = kick[env.joint_index['l_ankle_y']]

    # fast motions during swing
    trajectories = (
        ((5, init),),
        ((5, shift),),
        ((5, push),),
        ((.1, push_kick),),
        ((.1, kick),),
    )

    return trajectories

def linearly_interpolate(direct_trajectories, num_points):
    linear_trajectories = list(trajectories[:1])
    for t in range(1, len(trajectories)):

        _, source = trajectories[t-1][-1]
        duration, target = trajectories[t][-1]

        linear_trajectories.append([])
        for alpha in np.linspace(1, 0, num_points)[1:]:
            angles = alpha * source + (1 - alpha) * target
            linear_trajectories[t].append((duration / num_points, angles))

    return linear_trajectories

def extend_mirrored_trajectory(env, trajectories):

    # mirror for second step
    trajectories += tuple(
        tuple(
            (duration, env.mirror_position(angles))
            for (duration, angles) in trajectory)
        for trajectory in trajectories)

    return trajectories

def phase_trajectory_figure(env, trajectories):
    jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]
    for n, trajectory in enumerate(trajectories[1:]):
        pt.subplot(1, len(trajectories)-1, n+1)
        for t, (duration, angles) in enumerate(trajectory):
            jnt_loc = env.forward_kinematics(angles)
            jnt_loc -= jnt_loc[env.joint_index['r_toe']]
            render_legs(env, jnt_loc, jnt_idx, zoffset=2*t, alpha = (t+1) / len(trajectory))
        pt.axis('equal')
        pt.axis('off')
    pt.show()

if __name__ == "__main__":

    show_traj = True
    run_traj = False
    num_cycles = 10

    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)
    waypoints = get_waypoints(env,
        # angle from vertical axis to flat leg in initial stance
        init_flat = .02*np.pi,
        # angle for abs_y joint in initial stance
        init_abs_y = np.pi/16,
        # angle from swing leg to vertical axis in shift stance
        shift_swing = .05*np.pi,
        # angle of torso towards support leg in shift stance
        shift_torso = np.pi/6, #np.pi/5.75,
        # angle from vertical axis to flat leg in push stance
        push_flat = -.02*np.pi,#-.05*np.pi,
        # angle from swing leg to vertical axis in push stance
        push_swing = -.08*np.pi,#-.01*np.pi,
    )
    # (..., (angles, oojl, error), ...)

    phase_waypoint_figure(env, waypoints)

    trajectories = get_direct_trajectories(env, waypoints)

    trajectories = linearly_interpolate(trajectories, num_points=10)
    if show_traj:
        phase_trajectory_figure(env, trajectories)

    trajectories = extend_mirrored_trajectory(env, trajectories)
    pypot_trajectories = tuple(map(env.get_pypot_trajectory, trajectories))
    with open('pypot_traj1.pkl', "wb") as f: pk.dump(pypot_trajectories, f, protocol=2) 

    env.close()

    if run_traj:

        env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)
        env.settle(waypoints[0][0], seconds=3)
        input('...')

        for cycle in range(num_cycles):
            for trajectory in trajectories[(1 if cycle == 0 else 0):]:
                for (duration, angles) in trajectory:
                    env.goto_position(angles, duration=duration)
                # input('...')
    
        env.close()
