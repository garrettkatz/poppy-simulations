import os, sys
import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)
    base0 = env.get_base()
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = -90,
        cameraPitch = 0,
        cameraTargetPosition = base0[0],
    )
    MX64IDX = [env.joint_index[name] for name in ['abs_x', 'abs_y', 'l_hip_y', 'r_hip_y']]
    MX28IDX = [env.joint_index[name] for name in [
        'r_hip_x', 'r_hip_z', 'r_knee_y', 'r_ankle_y', 'l_hip_x', 'l_hip_z', 'l_knee_y', 'l_ankle_y', 'abs_z', 'bust_y', 'bust_x', 'head_z', 'head_y', 'l_shoulder_y', 'l_shoulder_x', 'l_arm_z', 'l_elbow_y', 'r_shoulder_y', 'r_shoulder_x', 'r_arm_z', 'r_elbow_y'
    ]]

    def get_pos():
        com_pos = np.zeros((env.num_joints, 3))
        jnt_pos = np.zeros((env.num_joints, 3))
        for idx in range(env.num_joints):
            state = pb.getLinkState(env.robot_id, idx)
            com_pos[idx] = state[0]
            jnt_pos[idx] = state[4]
        return com_pos, jnt_pos

    def show_support(com_pos, jnt_pos, names):
        mx64com = jnt_pos[MX64IDX].mean(axis=0)
        mx28com = jnt_pos[MX28IDX].mean(axis=0)    
        urdfcom = com_pos.mean(axis=0)
        support_polygon = np.array([jnt_pos[env.joint_index[name]] for name in names])
        pt.plot(support_polygon[:,0], support_polygon[:,1], 'ko-')
        pt.plot(urdfcom[0], urdfcom[1], 'ro')
        pt.plot(mx64com[0], mx64com[1], 'go')
        pt.plot(mx28com[0], mx28com[1], 'bo')
        pt.text(urdfcom[0], urdfcom[1], 'urdfcom')
        pt.text(mx64com[0], mx64com[1], 'mx64com')
        pt.text(mx28com[0], mx28com[1], 'mx28com')
        for name in names[:-1]:
            idx = env.joint_index[name]
            pt.text(jnt_pos[idx,0], jnt_pos[idx,1], name)            
        pt.axis('equal')

    ### PARAMS

    # half angle between upper legs and y-axis in initial stance
    init_hip_y = .025*np.pi
    # init_hip_y = 0

    # angle between front leg and y-axis in push stance
    push_front = -.02*np.pi

    # angle between back leg and y-axis in push stance
    push_back = .1*np.pi

    #### ZERO

    zero_angles = np.zeros(env.num_joints)
    env.set_position(zero_angles)
    for _ in range(240*1): env.step(zero_angles)
    zero_com_pos, zero_jnt_pos = get_pos()
    zero_base = env.get_base()

    show_support(zero_com_pos, zero_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
    print("ZERO")
    pt.show()

    #### INIT

    init_angles = np.zeros(env.num_joints)
    init_angles[env.joint_index['r_hip_y']] = -init_hip_y
    init_angles[env.joint_index['l_hip_y']] = +init_hip_y
    init_angles[env.joint_index['r_ankle_y']] = +init_hip_y
    init_angles[env.joint_index['l_ankle_y']] = -init_hip_y

    env.set_base(*zero_base)
    env.set_position(init_angles)
    for _ in range(240*1): env.step(init_angles)
    init_com_pos, init_jnt_pos = get_pos()
    init_base = env.get_base()

    # translational offset from front to back toes in init stance
    toe_to_toe = init_jnt_pos[env.joint_index['l_toe']] - init_jnt_pos[env.joint_index['r_toe']]

    # translational offset from back to front heels in init stance
    heel_to_heel = init_jnt_pos[env.joint_index['r_heel']] - init_jnt_pos[env.joint_index['l_heel']]
    heel_to_heel[0] *= -1 # reflect through yz plane since back foot becomes front
    print(f"front heel {init_jnt_pos[env.joint_index['r_heel']]}")
    print(f"back heel {init_jnt_pos[env.joint_index['l_heel']]}")
    print(f"heel to heel {heel_to_heel}")

    # urdfcom = init_com_pos.mean(axis=0)
    # pb.addUserDebugPoints(
    #     pointPositions = (urdfcom,),
    #     pointColorsRGB = ((1,0,0),),
    #     pointSize=40,
    # )
    # pb.addUserDebugText(
    #     "COM",
    #     textPosition = urdfcom,
    #     textColorRGB = (0,0,0),
    #     textSize=1,
    # )
    # print(f"urdf com: {urdfcom}")

    show_support(init_com_pos, init_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
    print("INIT")
    pt.show()

    #### PUSH

    # get target positions for most links
    push_angles = np.zeros(env.num_joints)
    push_angles[env.joint_index['r_hip_y']] = -push_front
    push_angles[env.joint_index['l_hip_y']] = +push_front
    push_angles[env.joint_index['r_ankle_y']] = +push_front
    push_angles[env.joint_index['l_ankle_y']] = -push_front

    env.set_base(*zero_base)
    env.set_position(push_angles)
    for _ in range(240*1): env.step(push_angles)
    push_com_pos, push_jnt_pos = get_pos()
    push_base = env.get_base()
    target_positions = push_jnt_pos.copy()

    # update back toe targe
    target_positions[env.joint_index['l_toe']] = push_jnt_pos[env.joint_index['r_toe']] + toe_to_toe

    # update back knee target
    push_angles[env.joint_index['l_hip_y']] = -push_back
    env.set_position(push_angles)
    knee_idx = env.joint_index['l_knee_y']
    target_positions[knee_idx] = pb.getLinkState(env.robot_id, knee_idx)[4]

    # run IK for final angles
    link_indices = [i for i in range(env.num_joints) if env.joint_name[i] not in ['l_heel', 'l_ankle_y']]
    target_positions = target_positions[link_indices]
    push_angles = env.inverse_kinematics(link_indices, target_positions)
    for name in ['l_ankle_y', 'l_knee_y']:
        print(f"{name}: {push_angles[env.joint_index[name]]}")

    env.set_base(*push_base)
    env.set_position(push_angles)
    for _ in range(240*1): env.step(push_angles)
    push_com_pos, push_jnt_pos = get_pos()
    # for _ in range(240*1): env.step(push_angles)

    show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
    print("PUSH")
    pt.show()

    ### SWING

    target_positions = push_jnt_pos.copy()

    # update (new) front heel
    heel_idx = env.joint_index['l_heel']
    target_positions[heel_idx] = push_jnt_pos[env.joint_index['r_heel']] + heel_to_heel
    print(f"target heel {target_positions[heel_idx]}")

    # run IK for final angles
    link_indices = [i for i in range(env.num_joints) if env.joint_name[i] not in ['l_toe', 'l_ankle_y']]
    target_positions = target_positions[link_indices]
    swing_angles = env.inverse_kinematics(link_indices, target_positions, num_iters=2000)
    for name in ['l_ankle_y', 'l_knee_y']:
        print(f"{name}: {swing_angles[env.joint_index[name]]}")

    env.set_base(*push_base)
    env.set_position(swing_angles)
    for _ in range(240*1): env.step(swing_angles)
    swing_com_pos, swing_jnt_pos = get_pos()
    # for _ in range(240*1): env.step(swing_angles)
    print(f"back heel {swing_jnt_pos[env.joint_index['r_heel']]}")
    print(f"front heel {swing_jnt_pos[heel_idx]}")

    show_support(swing_com_pos, swing_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'r_toe'])
    print("SWING")
    pt.show()

