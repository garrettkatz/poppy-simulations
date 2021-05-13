from poppy_env import PoppyEnv
from ZMPWalkPattern import ZMPWalkPatternGenerator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    # 
    zc = 0.3
    sx = 0.12
    sy = 0.065
    fh = 0.08
    dT = 5e-3
    num_steps = 8

    env = PoppyEnv(dT = dT)
    generator = ZMPWalkPatternGenerator(CoM_height = zc, foot_height = fh, shift_x = sx, shift_y = sy, dT = dT)
    CoMs, pred_ZMPs, ref_ZMPs, lfoot_traj, rfoot_traj = generator.generate(num_steps)

    rfoot_pos_list, lfoot_pos_list = [], []

    rjoints = env.inverse_kinematics(rfoot_traj[0] - CoMs[0], is_left = False)
    ljoints = env.inverse_kinematics(lfoot_traj[0] - CoMs[0], is_left = True)
    print(rjoints)
    print(ljoints)
    env.initialize(ljoints, rjoints)

    for com, lf, rf in zip(CoMs, lfoot_traj, rfoot_traj):

        rjoints = env.inverse_kinematics(rf - com, is_left = False)
        ljoints = env.inverse_kinematics(lf - com, is_left = True)

        env.setRightJointPositions(rjoints)
        env.setLeftJointPositions(ljoints)

        env.step()

        _, rfoot_pos = env.forward_kinematics(rjoints, False) 
        _, lfoot_pos = env.forward_kinematics(ljoints, True)
        rfoot_pos_list.append(rfoot_pos + com)
        lfoot_pos_list.append(lfoot_pos + com)
    
    rfoot_pos_list = np.array(rfoot_pos_list)
    lfoot_pos_list = np.array(lfoot_pos_list)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.scatter(lfoot_pos_list[:, 0], lfoot_pos_list[:, 1], lfoot_pos_list[:, 2], label = 'LF')
    # ax.scatter(rfoot_pos_list[:, 0], rfoot_pos_list[:, 1], rfoot_pos_list[:, 2], label = 'RF')
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.scatter(CoMs[:, 0], CoMs[:, 1], CoMs[:, 2], label = 'CoM')
    # ax.scatter(lfoot_traj[:, 0], lfoot_traj[:, 1], lfoot_traj[:, 2], label = 'LF')
    # ax.scatter(rfoot_traj[:, 0], rfoot_traj[:, 1], rfoot_traj[:, 2], label = 'RF')
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.scatter(pred_ZMPs[:, 0], pred_ZMPs[:, 1], pred_ZMPs[:, 2], label = 'pred')
    # ax.scatter(ref_ZMPs[:, 0], ref_ZMPs[:, 1], ref_ZMPs[:, 2], label = 'ref')
    # plt.legend()
    # plt.show()