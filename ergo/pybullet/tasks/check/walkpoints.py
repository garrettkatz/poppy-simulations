import os, sys
import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

if __name__ == "__main__":

    do_show = True

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=do_show)
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

    def iksolve(links, targets, angles, free):
        # constrain given links to target positions
        # and joints to given angles (except parent joints of free links)

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
        angles = env.inverse_kinematics(all_links, all_targets)

        # restore given angles
        env.set_position(save_angles)

        # return result
        return angles

    def reflectx(vec):
        return vec * np.array([-1, +1, +1]) # reflect through yz plane for when back foot becomes front

    def settle(angles, base=None, seconds=1):
        if base is not None: env.set_base(*base)
        env.set_position(angles)
        for _ in range(int(240*seconds)): env.step(angles)
        com_pos, jnt_pos = get_pos()
        base = env.get_base()
        return com_pos, jnt_pos, base

    def get_waypoints(
        # angle from y-axis to front leg in initial stance
        init_hip_y,
        # angle from y-axis to front leg in push stance
        push_front,
        # angle from back leg to y-axis in push stance
        push_back,
        # whether to visualize
        show,
    ):

        #### ZERO
    
        zero_angles = np.zeros(env.num_joints)
        zero_com_pos, zero_jnt_pos, zero_base = settle(zero_angles)

        print("ZERO")
        if show:    
            show_support(zero_com_pos, zero_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
            pt.show()
    
        #### INIT
    
        init_angles = np.zeros(env.num_joints)
        init_angles[env.joint_index['r_hip_y']] = -init_hip_y
        init_angles[env.joint_index['l_hip_y']] = +init_hip_y
        init_angles[env.joint_index['r_ankle_y']] = +init_hip_y
        init_angles[env.joint_index['l_ankle_y']] = -init_hip_y
        init_com_pos, init_jnt_pos, init_base = settle(init_angles, base=zero_base)
    
        # translational offset from front to back toes in init stance
        toe_to_toe = init_jnt_pos[env.joint_index['l_toe']] - init_jnt_pos[env.joint_index['r_toe']]
    
        # translational offset from back to front heels in init stance
        heel_to_heel = reflectx(init_jnt_pos[env.joint_index['r_heel']] - init_jnt_pos[env.joint_index['l_heel']])
        print(f"front heel {init_jnt_pos[env.joint_index['r_heel']]}")
        print(f"back heel {init_jnt_pos[env.joint_index['l_heel']]}")
        print(f"heel to heel {heel_to_heel}")
    
        print("INIT")
        if show:
            show_support(init_com_pos, init_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
            pt.show()
    
        #### PUSH

        # get target positions for most links
        push_angles = np.zeros(env.num_joints)
        push_angles[env.joint_index['r_hip_y']] = -push_front
        push_angles[env.joint_index['l_hip_y']] = +push_front
        push_angles[env.joint_index['r_ankle_y']] = +push_front
        push_angles[env.joint_index['l_ankle_y']] = -push_front

        push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base)

        # set up back upper leg angle
        push_angles[env.joint_index['l_hip_y']] = push_back

        # set up back toe target
        links = [env.joint_index['l_toe']]
        targets = (push_jnt_pos[env.joint_index['r_toe']] + toe_to_toe)[np.newaxis]
        free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
        push_angles = iksolve(links, targets, push_angles, free)
        for name in ['l_ankle_y', 'l_knee_y']:
            print(f"{name}: {push_angles[env.joint_index[name]]}")
        push_com_pos, push_jnt_pos, push_base = settle(push_angles, push_base)

        print("PUSH")
        if show:
            show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
            pt.show()
    
        ### SWING

        # set up back toe target
        links = [env.joint_index['l_heel']]
        targets = (push_jnt_pos[env.joint_index['r_heel']] + heel_to_heel)[np.newaxis]
        free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
        swing_angles = iksolve(links, targets, push_angles, free_joints)
        for name in ['l_ankle_y', 'l_toe']:
            print(f"{name}: {push_angles[env.joint_index[name]]}")
        swing_com_pos, swing_jnt_pos, swing_base = settle(swing_angles, push_base)

        print(f"back heel {swing_jnt_pos[env.joint_index['r_heel']]}")
        print(f"front heel {swing_jnt_pos[env.joint_index['l_heel']]}")
    
        print("SWING")
        if show:
            show_support(swing_com_pos, swing_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'r_toe'])
            pt.show()

        return (init_angles, push_angles, swing_angles)

    def get_traj(
        # angle from y-axis to front leg in initial stance
        init_front,
        # angle from back leg to y-axis in shift stance
        shift_back,
        # angle from y-axis to front leg in push stance
        push_front,
        # angle from back leg to y-axis in push stance
        push_back,
        # number of waypoints between each stance
        num_waypoints,
        # whether to visualize
        show,
    ):

        #### ZERO
    
        zero_angles = np.zeros(env.num_joints)
        zero_com_pos, zero_jnt_pos, zero_base = settle(zero_angles)

        print("ZERO")
        if show:    
            show_support(zero_com_pos, zero_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
            pt.show()
    
        #### INIT
    
        init_angles = np.zeros(env.num_joints)
        init_angles[env.joint_index['r_hip_y']] = -init_front
        init_angles[env.joint_index['l_hip_y']] = +init_front
        init_angles[env.joint_index['r_ankle_y']] = +init_front
        init_angles[env.joint_index['l_ankle_y']] = -init_front
        init_com_pos, init_jnt_pos, init_base = settle(init_angles, base=zero_base)
    
        # translational offset from back to front toes/heels in init stance
        toe_to_toe = init_jnt_pos[env.joint_index['r_toe']] - init_jnt_pos[env.joint_index['l_toe']]
        heel_to_heel = init_jnt_pos[env.joint_index['r_heel']] - init_jnt_pos[env.joint_index['l_heel']]
        print(f"front heel {init_jnt_pos[env.joint_index['r_heel']]}")
        print(f"back heel {init_jnt_pos[env.joint_index['l_heel']]}")
        print(f"heel to heel {heel_to_heel}")
    
        print("INIT")
        if show:
            show_support(init_com_pos, init_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
            pt.show()

        #### SHIFT
        shift_traj = np.empty((num_waypoints, env.num_joints))
        for t,w in enumerate(np.linspace(0, 1, num_waypoints)):

            shift_back_t = w*shift_back + (1-w)*init_front

            # get target angles for most links
            shift_angles = np.zeros(env.num_joints)
            shift_angles[env.joint_index['r_hip_y']] = -shift_back_t
            shift_angles[env.joint_index['l_hip_y']] = +shift_back_t
            shift_angles[env.joint_index['r_ankle_y']] = +shift_back_t
            shift_angles[env.joint_index['l_ankle_y']] = -shift_back_t
            shift_com_pos, shift_jnt_pos, shift_base = settle(shift_angles, zero_base)

            # set up front toe/heel target
            links = [env.joint_index[name] for name in ['r_toe','r_heel']]
            targets = np.stack((
                shift_jnt_pos[env.joint_index['l_toe']] + toe_to_toe,
                shift_jnt_pos[env.joint_index['l_heel']] + heel_to_heel,
            ))
            free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
            shift_angles = iksolve(links, targets, shift_angles, free)

            shift_traj[t] = shift_angles
            shift_com_pos, shift_jnt_pos, shift_base = settle(shift_angles, zero_base, seconds=.1)

            print("SHIFT")
            if show:
                show_support(shift_com_pos, shift_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()    

        #### PUSH

        push_traj = np.empty((num_waypoints, env.num_joints))
        for t,w in enumerate(np.linspace(0, 1, num_waypoints)):

            push_front_t = w*push_front + (1-w)*init_front
            push_back_t = w*push_back + (1-w)*init_front

            # get target angles for most links
            push_angles = np.zeros(env.num_joints)
            push_angles[env.joint_index['r_hip_y']] = -push_front_t
            push_angles[env.joint_index['l_hip_y']] = +push_front_t
            push_angles[env.joint_index['r_ankle_y']] = +push_front_t
            push_angles[env.joint_index['l_ankle_y']] = -push_front_t
            push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base)

            # set up back upper leg angle
            push_angles[env.joint_index['l_hip_y']] = push_back_t
    
            # set up back toe target
            links = [env.joint_index['l_toe']]
            targets = (push_jnt_pos[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
            free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
            push_angles = iksolve(links, targets, push_angles, free)

            push_traj[t] = push_angles
            push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base, seconds=.1)

            print("PUSH")
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
        # ### SWING

        # # set up back toe target
        # links = [env.joint_index['l_heel']]
        # targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
        # free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
        # swing_angles = iksolve(links, targets, push_angles, free_joints)
        # for name in ['l_ankle_y', 'l_toe']:
        #     print(f"{name}: {push_angles[env.joint_index[name]]}")
        # swing_com_pos, swing_jnt_pos, swing_base = settle(swing_angles, push_base)

        # print(f"back heel {swing_jnt_pos[env.joint_index['r_heel']]}")
        # print(f"front heel {swing_jnt_pos[env.joint_index['l_heel']]}")
    
        # print("SWING")
        # if show:
        #     show_support(swing_com_pos, swing_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'r_toe'])
        #     pt.show()

        # return (init_angles, push_angles, swing_angles)

        return push_traj

    # # get waypoints
    # init_angles, push_angles, swing_angles = get_waypoints(
    #     # angle from y-axis to front leg in initial stance
    #     init_hip_y = .025*np.pi,
    #     # angle from y-axis to front leg in push stance
    #     push_front = -.02*np.pi,
    #     # angle from back leg to y-axis in push stance
    #     push_back = -.1*np.pi,
    #     show = do_show,
    # )

    # get waypoints
    push_traj = get_traj(
        # angle from y-axis to front leg in initial stance
        init_front = .025*np.pi,
        # angle from back leg to y-axis in shift stance
        shift_back = .2*np.pi,
        # angle from y-axis to front leg in push stance
        push_front = -.02*np.pi,
        # angle from back leg to y-axis in push stance
        push_back = -.1*np.pi,
        num_waypoints = 10,
        show = do_show,
    )
    
    env.close()

    # visualize traj
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)
    settle(push_traj[0])
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = -90,
        cameraPitch = 0,
        cameraTargetPosition = env.get_base()[0],
    )

    input('.')
    for t in range(1, len(push_traj)):
        env.goto_position(push_traj[t], speed=1, duration=None, hang=False)
        input('.')

