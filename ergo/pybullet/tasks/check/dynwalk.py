import os, sys
import pickle as pk
import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

if __name__ == "__main__":

    do_show = True

    init_angle = np.pi/30

    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=do_show)
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = +90,
        cameraPitch = 0,
        cameraTargetPosition = env.get_base()[0],
    )

    def get_pos():
        com_pos = np.zeros((env.num_joints, 3))
        jnt_pos = np.zeros((env.num_joints, 3))
        for idx in range(env.num_joints):
            state = pb.getLinkState(env.robot_id, idx)
            com_pos[idx] = state[0]
            jnt_pos[idx] = state[4]
        return com_pos, jnt_pos

    def settle(angles, base=None, seconds=1):
        if base is not None: env.set_base(*base)
        env.set_position(angles)
        for _ in range(int(240*seconds)): env.step(angles)
        com_pos, jnt_pos = get_pos()
        base = env.get_base()
        return com_pos, jnt_pos, base

    def iksolve(links, targets, angles, free, num_iters=1000, verbose=False):
        # constrain given links to target positions
        # and joints to given angles (except parent joints of free links)
        # return full solution joints, out-of-range flag, and constraint error

        # save current angles for temporary overwrite
        save_angles = env.get_position()

        # get targets for all other non-free joints
        env.set_position(angles)
        all_targets = get_pos()[1]
        all_targets[links] = targets
        all_links = np.array([j for j in range(env.num_joints) if j not in free])
        assert all([l in all_links for l in links]) # make sure original targets are not free links
        all_targets = all_targets[all_links]

        # IK to solve free joints
        angles = env.inverse_kinematics(all_links, all_targets, num_iters=num_iters)

        # guard against occasional out-of-joint-range solutions
        # assert(all([
        #     (env.joint_low[i] <= angles[i] <= env.joint_high[i]) or env.joint_fixed[i]
        #     for i in range(env.num_joints)]))
        oojr = False
        for i in range(env.num_joints):
            if not ((env.joint_low[i] <= angles[i] <= env.joint_high[i]) or env.joint_fixed[i]):
                print(f"{env.joint_name[i]}: {angles[i]} not in [{env.joint_low[i]}, {env.joint_high[i]}]!")
                oojr = True
        if oojr: input('uh oh...')

        # measure constraint error
        env.set_position(angles)
        all_actual = get_pos()[1]
        all_actual = all_actual[all_links]
        error = np.fabs(all_targets - all_actual).max()

        if verbose:
            print('iksolve errors:')
            print([env.joint_name[i] for i in range(env.num_joints) if i not in free])
            print(all_targets - all_actual)
            print(error)

        # restore given angles
        env.set_position(save_angles)

        # return result
        return angles, oojr, error

    # get base in zero joint angle pose
    zero_angles = np.zeros(env.num_joints)
    _, _, zero_base = settle(zero_angles)

    # initial waypoint pose
    init_angles = env.angle_array({
        'r_hip_y': -init_angle,
        'l_hip_y': +init_angle,
        'r_ankle_y': +init_angle,
        'l_ankle_y': -init_angle,
    }, convert=False)
    _, init_jnt_pos, _ = settle(init_angles, base=zero_base, seconds=0)

    # translational offset from back to front toes/heels in init stance
    foot_to_foot = init_jnt_pos[env.joint_index['r_toe']] - init_jnt_pos[env.joint_index['l_toe']]

    # push pose
    push_angles = env.angle_array({
        'l_hip_y': -2*init_angle,
    }, convert=False)
    _, push_jnt_pos, _ = settle(push_angles, base=zero_base, seconds=0)

    # set up back toe target
    links = [env.joint_index['l_toe']]
    targets = (push_jnt_pos[env.joint_index['r_toe']] - foot_to_foot)[np.newaxis]
    free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
    push_angles, push_oojr, push_error = iksolve(links, targets, push_angles, free, num_iters=2000)
    push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base, seconds=0)

    # try pushing, see what happens
    settle(init_angles, base=zero_base, seconds=1)
    input('push...')
    while True:
        env.step(push_angles)
        input('.')

