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
    linear_trajectories = []
    for t in range(len(direct_trajectories)):

        _, source = direct_trajectories[t][-1]
        duration, target = direct_trajectories[(t+1) % len(direct_trajectories)][-1]
        if (t + 1) == len(direct_trajectories):
            target = env.mirror_position(target)

        linear_trajectories.append([])
        for alpha in np.linspace(1, 0, num_points):
            angles = alpha * source + (1 - alpha) * target
            linear_trajectories[t].append((duration / num_points, angles))

    return linear_trajectories

def constrained_interpolate(direct_trajectories, num_points):
    constrained_trajectories = []
    for t in range(len(direct_trajectories)):

        _, source = direct_trajectories[t][-1]
        duration, target = direct_trajectories[(t+1) % len(direct_trajectories)][-1]
        if (t + 1) == len(direct_trajectories):
            target = env.mirror_position(target)

        # translational offset from back to front toes/heels in target stance
        jnt_loc = env.forward_kinematics(target)
        toe_to_toe = jnt_loc[env.joint_index['r_toe']] - jnt_loc[env.joint_index['l_toe']]
        heel_to_heel = jnt_loc[env.joint_index['r_heel']] - jnt_loc[env.joint_index['l_heel']]

        constrained_trajectories.append([])
        for a, alpha in enumerate(np.linspace(1, 0, num_points)):
            angles = alpha * source + (1 - alpha) * target

            # enforce constraints
            jnt_loc = env.forward_kinematics(angles)
            if t == 0: # shift to push
                links = [env.joint_index['r_toe'], env.joint_index['r_heel']]
                targets = np.stack((
                    jnt_loc[env.joint_index['l_toe']] + toe_to_toe,
                    jnt_loc[env.joint_index['l_heel']] + heel_to_heel,
                ))
                free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)

            if t == 1: # shift to push
                links = [env.joint_index['l_toe']]
                targets = (jnt_loc[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)

            if t == 4: # kick to mirrored init
                links = [env.joint_index['l_heel']]
                targets = (jnt_loc[env.joint_index['r_heel']] - heel_to_heel)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000, resid_thresh=1e-7, verbose=False)

            # last of t is first of t+1 so first duration is 0
            dur = 0 if a == 0 else (duration / (num_points - 1))

            constrained_trajectories[t].append((dur, angles))

    return constrained_trajectories

def extend_mirrored_trajectory(env, trajectories):

    # mirror for second step
    trajectories += tuple(
        tuple(
            (duration, env.mirror_position(angles))
            for (duration, angles) in trajectory)
        for trajectory in trajectories)

    return trajectories

def phase_trajectory_figure(env, trajectories, fname=None):
    jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]
    for n, trajectory in enumerate(trajectories):
        pt.subplot(1, len(trajectories), n+1)
        for t, (duration, angles) in enumerate(trajectory):
            # if n == 0 and 0 < t < len(trajectory)-1: continue
            jnt_loc = env.forward_kinematics(angles)
            jnt_loc -= jnt_loc[env.joint_index['r_toe']]
            render_legs(env, jnt_loc, jnt_idx, zoffset=2*t, alpha = (t+1) / len(trajectory))
        pt.axis('equal')
        pt.axis('off')
    if fname is not None: pt.savefig(fname)
    pt.show()

if __name__ == "__main__":

    show_traj = True
    run_traj = True
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

    # trajectories = linearly_interpolate(trajectories, num_points=5)
    trajectories = constrained_interpolate(trajectories, num_points=5)
    if show_traj:
        phase_trajectory_figure(env, trajectories, fname='transitions.pdf')

    trajectories = extend_mirrored_trajectory(env, trajectories)
    pypot_trajectories = tuple(map(env.get_pypot_trajectory, trajectories))
    with open('pypot_traj1.pkl', "wb") as f: pk.dump(pypot_trajectories, f, protocol=2) 

    env.close()

    if run_traj:

        env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)
        env.settle(waypoints[0][0], seconds=2)
        input('...')

        for cycle in range(num_cycles):
            # for trajectory in trajectories[(1 if cycle == 0 else 0):]:
            for t, trajectory in enumerate(trajectories):
                # input(f"{t}")
                for (duration, angles) in trajectory:
                    env.goto_position(angles, duration=duration)
                    # input('..')
    
        env.close()
