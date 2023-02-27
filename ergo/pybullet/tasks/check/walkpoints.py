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

    do_synthesis = True
    do_gait_figs = False
    do_filter = False
    show_filter = False
    # do_show = False
    # num_waypoints = 20
    do_show = True
    num_waypoints = 3

    pt.rcParams["text.usetex"] = True
    pt.rcParams['font.family'] = 'serif'

    if do_synthesis:

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

        def get_init_waypoint(init_front, init_abs_y, zero_base):

            # initial waypoint pose
            init_angles = env.angle_array({
                'r_hip_y': -init_front,
                'l_hip_y': +init_front,
                'r_ankle_y': +init_front,
                'l_ankle_y': -init_front,
                'abs_y': init_abs_y,
            }, convert=False)
            _, init_jnt_pos, _ = env.settle(init_angles, base=zero_base, seconds=0)

            init_oojr = False
            init_error = 0

            # translational offset from back to front toes/heels in init stance
            foot_to_foot = init_jnt_pos[env.joint_index['r_toe']] - init_jnt_pos[env.joint_index['l_toe']]

            return init_angles, init_oojr, init_error, foot_to_foot

        def get_shift_waypoint(shift_back, shift_torso, zero_base, foot_to_foot):

            # shift waypoint pose
            shift_angles = env.angle_array({
                'r_hip_y': -shift_back,
                'l_hip_y': +shift_back,
                'r_ankle_y': +shift_back,
                'l_ankle_y': -shift_back,
                'abs_x': +shift_torso,
                'bust_x': -shift_torso,
                'l_shoulder_x': +shift_torso,
                'r_shoulder_x': -shift_torso,
                'r_knee_y': 0.1 # to discourage out-of-joint-range solutions
            }, convert=False)
            _, shift_jnt_pos, _ = env.settle(shift_angles, base=zero_base, seconds=0)

            # set up front toe/heel target
            links = [env.joint_index[name] for name in ['r_toe','r_heel']]
            targets = np.stack((
                shift_jnt_pos[env.joint_index['l_toe']] + foot_to_foot,
                shift_jnt_pos[env.joint_index['l_heel']] + foot_to_foot,
            ))
            free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
            shift_angles, shift_oojr, shift_error = env.partial_ik(links, targets, shift_angles, free, num_iters=2000)

            return shift_angles, shift_oojr, shift_error

        def get_push_waypoint(push_front, push_back, shift_torso, zero_base, foot_to_foot):

            # push stance
            push_angles = env.angle_array({
                'r_hip_y': -push_front,
                'l_hip_y': push_back,
                'r_ankle_y': +push_front,
                'abs_x': +shift_torso,
                'bust_x': -shift_torso,
                'l_shoulder_x': +shift_torso,
                'r_shoulder_x': -shift_torso,
            }, convert=False)
            _, push_jnt_pos, _ = env.settle(push_angles, zero_base, seconds=0)
    
            # set up back toe target
            links = [env.joint_index['l_toe']]
            targets = (push_jnt_pos[env.joint_index['r_toe']] - foot_to_foot)[np.newaxis]
            free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
            push_angles, push_oojr, push_error = env.partial_ik(links, targets, push_angles, free, num_iters=2000, verbose=False)
            # input(f"{push_angles[env.joint_index['l_hip_y']]} .")
            push_com_pos, push_jnt_pos, push_base = env.settle(push_angles, zero_base, seconds=0)

            return push_angles, push_oojr, push_error

        def get_kick_waypoint(push_angles, zero_base, foot_to_foot):

            _, push_jnt_pos, _ = env.settle(push_angles, zero_base, seconds=0)

            # set up heel target for swinging leg (reflect since back becomes front)
            links = [env.joint_index['l_heel']]
            targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(foot_to_foot))[np.newaxis]
            free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]#, 'l_knee_y']]
            kick_angles, kick_oojr, kick_error = env.partial_ik(links, targets, push_angles, free_joints, num_iters=3000, verbose=False)

            return kick_angles, kick_oojr, kick_error

        # right starts in front
        def get_waypoints(
            # angle from vertical axis to front leg in initial stance
            init_front,
            # angle for abs_y joint in initial stance
            init_abs_y,
            # angle from back leg to vertical axis in shift stance
            shift_back,
            # angle of torso towards support leg in shift stance
            shift_torso,
            # angle from vertical axis to front leg in push stance
            push_front,
            # angle from back leg to vertical axis in push stance
            push_back,
            # whether to visualize
            show,
        ):

            # get base in zero joint angle pose
            zero_angles = np.zeros(env.num_joints)
            _, _, zero_base = env.settle(zero_angles)

            init_angles, init_oojr, init_error, foot_to_foot = get_init_waypoint(init_front, init_abs_y, zero_base)
            shift_angles, shift_oojr, shift_error = get_shift_waypoint(shift_back, shift_torso, zero_base, foot_to_foot)
            push_angles, push_oojr, push_error = get_push_waypoint(push_front, push_back, shift_torso, zero_base, foot_to_foot)
            kick_angles, kick_oojr, kick_error = get_kick_waypoint(push_angles.copy(), zero_base, foot_to_foot)

            return (
                (init_angles, init_oojr, init_error),
                (shift_angles, shift_oojr, shift_error),
                (push_angles, push_oojr, push_error),
                (kick_angles, kick_oojr, kick_error),
            )

        def interpolate_waypoints(start_angles, end_angles, num_waypoints, links, free, get_targets):
            # at every trajectory timepoint, given links are constrained to targets while IK solves for free joints
            # get_targets(jnt_pos) should return targets for given joint position

            traj = np.empty((num_waypoints, env.num_joints))
            for t,w in enumerate((1 + np.cos(np.linspace(0, np.pi, num_waypoints)))/2):

                # interpolate angles and get current joint positions
                angles = w*start_angles + (1-w)*end_angles
                _, jnt_pos, _ = env.settle(angles, seconds=0)
    
                # solve for targets
                targets = get_targets(jnt_pos)
                angles, _, _ = env.partial_ik(links, targets, angles, free, num_iters=2000)

                # save trajectory
                traj[t] = shift_angles

            return traj

    
        def get_traj(
            # angle from vertical axis to front leg in initial stance
            init_front,
            # angle for abs_y joint in initial stance
            init_abs_y,
            # angle from back leg to vertical axis in shift stance
            shift_back,
            # angle of torso towards support leg in shift stance
            shift_torso,
            # angle from vertical axis to left (swing) leg in push stance
            push_front,
            # angle from right (support) leg to vertical axis in push stance
            push_back,
            # number of waypoints between each stance
            num_waypoints,
            # whether to visualize
            show,
        ):
    
            #### ZERO
        
            zero_angles = np.zeros(env.num_joints)
            zero_com_pos, zero_jnt_pos, zero_base = env.settle(zero_angles)
    
            print("ZERO")
            if show:    
                show_support(zero_com_pos, zero_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'l_ankle_y', 'l_toe', 'r_toe'])
                pt.show()
        
            #### INIT
            # init_abs_y = np.pi/16
        
            init_angles = np.zeros(env.num_joints)
            init_angles[env.joint_index['r_hip_y']] = -init_front
            init_angles[env.joint_index['l_hip_y']] = +init_front
            init_angles[env.joint_index['r_ankle_y']] = +init_front
            init_angles[env.joint_index['l_ankle_y']] = -init_front
            # init_angles[env.joint_index['r_shoulder_x']] = -np.pi/8
            # init_angles[env.joint_index['l_shoulder_x']] = +np.pi/8
            init_angles[env.joint_index['abs_y']] = init_abs_y
            init_com_pos, init_jnt_pos, init_base = env.settle(init_angles, base=zero_base, seconds=2)
        
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
            # shift_torso = np.pi/5.75
            # for t,w in enumerate(np.linspace(0, 1, num_waypoints)):
            for t,w in enumerate(.5 - .5*np.cos(np.linspace(0, np.pi, num_waypoints))):
    
                shift_back_t = w*shift_back + (1-w)*init_front
                # shift_shoulder_t = w*(-np.pi/8) + (1-w)*init_angles[env.joint_index['r_shoulder_x']]
                shift_abs_y_t = w*0 + (1-w)*init_angles[env.joint_index['abs_y']]
                shift_abs_x_t = w*shift_torso + (1-w)*init_angles[env.joint_index['abs_x']]
                shift_bust_x_t = -w*shift_torso + (1-w)*init_angles[env.joint_index['bust_x']]
                shift_shoulder_x_t = w*shift_torso + (1-w)*init_angles[env.joint_index['l_shoulder_x']]
    
                # get target angles for most links
                shift_angles = np.zeros(env.num_joints)
                shift_angles[env.joint_index['r_hip_y']] = -shift_back_t
                shift_angles[env.joint_index['l_hip_y']] = +shift_back_t
                shift_angles[env.joint_index['r_ankle_y']] = +shift_back_t
                shift_angles[env.joint_index['l_ankle_y']] = -shift_back_t
                shift_angles[env.joint_index['abs_y']] = shift_abs_y_t
                # shift_angles[env.joint_index['r_shoulder_x']] = shift_shoulder_t

                shift_angles[env.joint_index['abs_x']] = shift_abs_x_t
                shift_angles[env.joint_index['bust_x']] = shift_bust_x_t
                shift_angles[env.joint_index['l_shoulder_x']] = +shift_shoulder_x_t
                shift_angles[env.joint_index['r_shoulder_x']] = -shift_shoulder_x_t

                shift_com_pos, shift_jnt_pos, shift_base = env.settle(shift_angles, zero_base, seconds=1)
    
                # initialize positive knee angle to avoid out-of-joint-range solutions
                shift_angles[env.joint_index['r_knee_y']] = 0.1
    
                # set up front toe/heel target
                links = [env.joint_index[name] for name in ['r_toe','r_heel']]
                targets = np.stack((
                    shift_jnt_pos[env.joint_index['l_toe']] + toe_to_toe,
                    shift_jnt_pos[env.joint_index['l_heel']] + heel_to_heel,
                ))
                free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
                shift_angles, _, _ = env.partial_ik(links, targets, shift_angles, free, num_iters=2000)
    
                env.set_position(shift_angles)
    
                shift_traj[t] = shift_angles
                shift_com_pos, shift_jnt_pos, shift_base = env.settle(shift_angles, zero_base, seconds=1)
    
                print("SHIFT", t)
            if show:
                show_support(shift_com_pos, shift_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()    
    
            final_shift_angles = shift_angles
    
            #### PUSH
    
            ## solve final push pose first then interpolate since both knees bent during motion
    
            # get target angles for most links
            push_angles = np.zeros(env.num_joints)
            # push_angles[env.joint_index['r_hip_y']] = -push_front
            # push_angles[env.joint_index['l_hip_y']] = +push_front
            # push_angles[env.joint_index['r_ankle_y']] = +push_front
            # push_angles[env.joint_index['l_ankle_y']] = -push_front
            # # push_angles[env.joint_index['r_shoulder_x']] = -np.pi/8
            # # push_angles[env.joint_index['l_shoulder_x']] = +np.pi/8
            push_angles[env.joint_index['r_hip_y']] = -push_back
            push_angles[env.joint_index['l_hip_y']] = +push_back
            push_angles[env.joint_index['r_ankle_y']] = +push_back
            push_angles[env.joint_index['l_ankle_y']] = -push_back

            push_angles[env.joint_index['abs_x']] = final_shift_angles[env.joint_index['abs_x']]
            push_angles[env.joint_index['bust_x']] = final_shift_angles[env.joint_index['bust_x']]
            push_angles[env.joint_index['l_shoulder_x']] = final_shift_angles[env.joint_index['l_shoulder_x']]
            push_angles[env.joint_index['r_shoulder_x']] = final_shift_angles[env.joint_index['r_shoulder_x']]

            push_com_pos, push_jnt_pos, push_base = env.settle(push_angles, zero_base)
    
            # # set up back upper leg angle
            # push_angles[env.joint_index['l_hip_y']] = push_back
            # set up front upper leg angle
            push_angles[env.joint_index['l_hip_y']] = push_front
    
            # # initialize positive knee angle to avoid out-of-joint-range solutions
            # push_angles[env.joint_index['l_knee_y']] = 0.5
    
            # set up back toe target
            links = [env.joint_index['l_toe']]
            targets = (push_jnt_pos[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
            free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
            final_push_angles, _, _ = env.partial_ik(links, targets, push_angles, free, num_iters=2000)
    
            push_com_pos, push_jnt_pos, push_base = env.settle(final_push_angles, zero_base)#, seconds=0.1)
            print("PUSH final")
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
            push_traj = np.empty((num_waypoints, env.num_joints))
            # for t,w in enumerate(np.linspace(0, 1, num_waypoints)):
            for t,w in enumerate(.5 - .5*np.cos(np.linspace(0, np.pi, num_waypoints))):
    
                # get link positions at interpolated waypoint
                push_angles = w*final_push_angles + (1-w)*final_shift_angles
                _, push_jnt_pos, _ = env.settle(push_angles, zero_base, seconds=0)
    
                # set up back toe target
                links = [env.joint_index['l_toe']]
                targets = (push_jnt_pos[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
    
                # re-solve knee/ankle to satisfy toe constraint
                push_angles, _, _ = env.partial_ik(links, targets, push_angles, free, num_iters=2000, verbose=False)
                env.set_position(push_angles)
    
                push_traj[t] = push_angles
                push_com_pos, push_jnt_pos, push_base = env.settle(push_angles, zero_base, seconds=1)
    
                print("PUSH", t)
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
            ### KICK
    
            # set up heel target for swinging leg (reflect since back becomes front)
            links = [env.joint_index['l_heel']]
            targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
            free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]#, 'l_knee_y']]
            swing_angles, _, _ = env.partial_ik(links, targets, push_angles, free_joints, num_iters=3000, verbose=True)
            swing_com_pos, swing_jnt_pos, swing_base = env.settle(swing_angles, push_base, seconds=3)

            # change ankle before knee for better clearance
            swing_traj = np.stack((push_angles, swing_angles))
            swing_traj[0,env.joint_index['l_ankle_y']] = swing_angles[env.joint_index['l_ankle_y']]

            # # fast swing
            # swing_traj = swing_angles[np.newaxis,:]
    
            print("KICK")
            if show:
                show_support(swing_com_pos, swing_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'r_toe'])
                pt.show()
    
            # ### PLANT
    
            # set up final plant angles (reflection of init angles)
            final_plant_angles = np.zeros(env.num_joints)
            final_plant_angles[env.joint_index['r_hip_y']] = +init_front
            final_plant_angles[env.joint_index['l_hip_y']] = -init_front
            final_plant_angles[env.joint_index['r_ankle_y']] = -init_front
            final_plant_angles[env.joint_index['l_ankle_y']] = +init_front
            # final_plant_angles[env.joint_index['r_shoulder_x']] = -np.pi/8
            # final_plant_angles[env.joint_index['l_shoulder_x']] = +np.pi/8

            final_plant_angles[env.joint_index['abs_y']] = init_abs_y
    
            plant_traj = np.empty((num_waypoints, env.num_joints))
            # for t,w in enumerate(np.linspace(0, 1, num_waypoints)):
            for t,w in enumerate(.5 - .5*np.cos(np.linspace(0, np.pi, num_waypoints))):
    
                # get link positions at interpolated waypoint
                plant_angles = w*final_plant_angles + (1-w)*swing_angles
                _, plant_jnt_pos, _ = env.settle(plant_angles, zero_base, seconds=0)

                # set up heel target for planted leg (reflect since back becomes front)
                links = [env.joint_index['l_heel']]
                targets = (swing_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
                free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
                plant_angles, _, _ = env.partial_ik(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
                plant_com_pos, plant_jnt_pos, plant_base = env.settle(plant_angles, swing_base, seconds=1)
    
                # use knee position from interpolant, just re-solve ankle
                plant_angles, _, _ = env.partial_ik(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
                env.set_position(plant_angles)
    
                plant_traj[t] = plant_angles
                plant_com_pos, plant_jnt_pos, plant_base = env.settle(plant_angles, zero_base, seconds=1)
    
                print("PLANT", t)
            if show:
                show_support(plant_com_pos, plant_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_heel', 'r_toe'])
                pt.show()
    
            return (
                (shift_traj, .3),
                (push_traj, .3),
                (swing_traj, 4.),
                (plant_traj, .3),
            )

        #### MAIN

        ### do IK and CoM filter
        if do_filter or show_filter:

            grid_sampling = 10
            init_front_range = np.linspace(np.pi/100, np.pi/16, grid_sampling)
            init_abs_y = np.pi/16
            shift_back_range = np.linspace(np.pi/100, np.pi/4, grid_sampling)
            shift_torso = np.pi/5.75
            push_back_range = np.linspace(-np.pi/4, np.pi/10, grid_sampling)
            push_front_range = np.linspace(-np.pi/10, np.pi/10, grid_sampling)

            print('init_front_range:', init_front_range)
            print('shift_back_range:', shift_back_range)
            print('push_back_range:', push_back_range)
            print('push_front_range:', push_front_range)

        if do_filter:

            # get base in zero joint angle pose
            zero_angles = np.zeros(env.num_joints)
            _, _, zero_base = env.settle(zero_angles)
    
            foot_to_foot = np.empty((len(init_front_range), 3))
            init_angles = np.empty((len(init_front_range), env.num_joints), dtype=bool)
            init_oojr = np.empty(len(init_front_range), dtype=bool)
            for i, init_front in enumerate(init_front_range):
                print(f"init {i}/{len(init_front_range)}")
                init_angles[i], init_oojr[i], init_error, foot_to_foot[i] = get_init_waypoint(init_front, init_abs_y, zero_base)
    
            shift_oojr = np.zeros((len(foot_to_foot), len(shift_back_range)), dtype=bool)
            shift_error = -np.ones((len(foot_to_foot), len(shift_back_range)))
            shift_support = np.zeros((len(foot_to_foot), len(shift_back_range)), dtype=bool)
            for i, init_front in enumerate(init_front_range):
                for j, shift_back in enumerate(shift_back_range):
                    print(f"shift {i},{j}/{len(init_front_range)},{len(shift_back_range)}")
                    shift_angles, shift_oojr[i,j], shift_error[i,j] = get_shift_waypoint(shift_back, shift_torso, zero_base, foot_to_foot[i])
                    com_pos, jnt_pos, _ = env.settle(shift_angles, zero_base, seconds=0)
                    names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
                    shift_support[i,j] = within_support(com_pos, jnt_pos, names)

            swing_oojr = np.zeros((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            swing_error = -np.ones((len(foot_to_foot), len(push_front_range), len(push_back_range)))
            swing_support = np.zeros((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            swing_clear = np.ones((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            for i, init_front in enumerate(init_front_range):
                for j, push_front in enumerate(push_front_range):
                    for k, push_back in enumerate(push_back_range):
                        print(f"shift {i},{j},{k}/{len(init_front_range)},{len(push_front_range)},{len(push_back_range)}")
                        push_angles, push_oojr, push_error = get_push_waypoint(push_front, push_back, shift_torso, zero_base, foot_to_foot[i])
                        kick_angles, kick_oojr, kick_error = get_kick_waypoint(push_angles, zero_base, foot_to_foot[i])
                        swing_oojr[i,j,k] = push_oojr or kick_oojr
                        swing_error[i,j,k] = max(push_error, kick_error)

                        com_pos, jnt_pos, _ = env.settle(push_angles, zero_base, seconds=0)
                        push_support = within_support(com_pos, jnt_pos, ('r_toe', 'r_heel', 'l_toe', 'r_toe'))
                        com_pos, jnt_pos, _ = env.settle(kick_angles, zero_base, seconds=0)
                        kick_support = within_support(com_pos, jnt_pos, ('r_toe', 'r_heel', 'l_heel', 'r_toe'))
                        swing_support[i,j,k] = push_support and kick_support

                        # check clearance at 8 intermediate points
                        for w in np.linspace(0, 1, 10):
                            angles = (1-w)*push_angles + w*kick_angles
                            _, jnt_pos, _ = env.settle(angles, seconds=0)
                            clear_names = ()
                            if w > 0: clear_names += ('l_toe',)
                            if w < 1: clear_names += ('l_heel',)
                            for name in clear_names:
                                if jnt_pos[env.joint_index[name],2] <= 0:
                                    swing_clear[i,j,k] = False
                                    break
                            if not swing_clear[i,j,k]: break

            with open(f'filter_{grid_sampling}.pkl','wb') as f:
                pk.dump((shift_oojr, shift_error, shift_support, swing_oojr, swing_error, swing_support, swing_clear), f)

        if show_filter:

            with open(f'filter_{grid_sampling}.pkl','rb') as f:
                (shift_oojr, shift_error, shift_support, swing_oojr, swing_error, swing_support, swing_clear) = pk.load(f)

            fig, ax = pt.subplots(1, 3, figsize=(4,1.6), constrained_layout=True)
            ax[0].imshow(~shift_oojr, cmap='gray')
            ax[0].set_title('In Limits')
            ax[1].imshow(shift_support, cmap='gray')
            ax[1].set_title('CoM Support')
            im = ax[2].imshow(shift_error, vmin=0, cmap='gray')
            ax[2].set_title('IK Error')
            fig.colorbar(im, ax=ax[2])
            for a in ax:
                a.set_xticks([0.5, len(shift_back_range)-.5], [f"{shift_back_range[i]:.2f}" for i in [0, -1]])
                a.set_yticks([], [])
            ax[0].set_yticks([0.5, len(init_front_range)-.5], [f"{init_front_range[i]:.2f}" for i in [0, -1]])
            # fig.supxlabel("$\\theta^{(\\mathbf{S})}_\\ell$", fontsize=14)
            # fig.supylabel("$\\theta^{(\\mathbf{I})}_r$", rotation=0, fontsize=14)
            ax[1].set_xlabel("$\\theta^{(\\mathbf{S})}_\\ell$", fontsize=14)
            ax[0].set_ylabel("$\\theta^{(\\mathbf{I})}_\\ell$", rotation=0, fontsize=14)
            pt.savefig(f'shift_search_{grid_sampling}.eps')
            pt.show()

            fig, ax = pt.subplots(2, len(init_front_range), figsize=(13,3.5), constrained_layout=True)
            for i, init_front in enumerate(init_front_range):
                feas = (~swing_oojr[i] & swing_support[i]).astype(int) + swing_clear[i]
                ax[0,i].imshow(feas, vmax=2, cmap='gray')
                im_i = ax[1,i].imshow(swing_error[i], vmin=0, vmax=swing_error.max(),cmap='gray')
                if swing_error[i].max() == swing_error.max(): im = im_i
                if i == 0:
                    ax[0,i].set_yticks([0.5, len(push_front_range)-.5], [f"{push_front_range[k]:.2f}" for k in [0, -1]])
                    ax[1,i].set_yticks([0.5, len(push_front_range)-.5], [f"{push_front_range[k]:.2f}" for k in [0, -1]])
                else:
                    ax[0,i].set_yticks([], [])
                    ax[1,i].set_yticks([], [])
                ax[0,i].set_title(f"{init_front:.2f}")
                ax[0,i].set_xticks([], [])
                ax[1,i].set_xticks([0.5, len(push_back_range)-.5], [f"{push_back_range[k]:.2f}" for k in [0, -1]])
            ax[0,0].set_ylabel('Feasible')
            ax[1,0].set_ylabel('IK Error')
            fig.suptitle("$\\theta^{(\mathbf{I})}_r$")
            fig.supxlabel("$\\theta^{(\mathbf{P})}_\ell$")
            fig.supylabel("$\\theta^{(\mathbf{P})}_r$")
            fig.colorbar(im, ax=ax[:,-1])
            pt.savefig(f'swing_search_{grid_sampling}.eps')
            pt.show()

            # pt.subplot(1,2,1)
            # pt.hist(shift_error[~shift_oojr & shift_support], fc='w', ec='k')
            # pt.ylabel('Frequency')
            # pt.xlabel('IK error')
            # pt.title('Shift')
            # pt.subplot(1,2,2)
            # pt.hist(swing_error[~swing_oojr & swing_support], fc='w', ec='k')
            # pt.xlabel('IK error')
            # pt.title('Swing')
            # pt.show()
            

        if do_filter or show_filter:

            env.close()
            import sys
            sys.exit()
        
        # get trajectory
        # # worked once
        # trajs = get_traj(
        #     # angle from vertical axis to front leg in initial stance
        #     init_front = .02*np.pi,
        #     init_abs_y = np.pi/16,
        #     # angle from back leg to vertical axis in shift stance
        #     shift_back = .05*np.pi,
        #     shift_torso = np.pi/5.75,
        #     # angle from vertical axis to front leg in push stance
        #     push_front = -.05*np.pi,
        #     # angle from back leg to vertical axis in push stance
        #     push_back = -.01*np.pi,
        #     num_waypoints = num_waypoints,
        #     show = do_show,
        # )
        # from grid search
        trajs = get_traj(
            # angle from vertical axis to front leg in initial stance
            init_front = 0.03141593,#.15*np.pi,
            # angle for abs_y joint in initial stance
            init_abs_y = np.pi/16,
            # angle from back leg to vertical axis in shift stance
            shift_back = 0.13912767,#.05*np.pi,
            # angle of torso towards support leg in shift stance
            shift_torso = np.pi/10,
            # angle from vertical axis to front (right) leg in push stance
            push_front = 0.0448799,#-.05*np.pi,
            # angle from back (left) leg to vertical axis in push stance
            push_back = -0.5,#-0.15707963,#-.01*np.pi,
            num_waypoints = num_waypoints,
            # whether to visualize
            show = do_show,
        )
        env.close()
        with open('traj1.pkl', "wb") as f: pk.dump(trajs, f)

        # pypot-compatible format
        prev_angles = np.zeros(env.num_joints)
        trajs = ((trajs[0][0][:1], .5),) + trajs # first send to init angles
        pypot_trajs = []
        for mirror in (False, True): # pypot doesn't have env's mirror function, do it here
            for (traj, speed) in trajs:
                pypot_traj = []
                if len(traj) > 1:  traj = traj[1:] # last waypoint of one traj = first of next

                # duration for whole motion, not waypoints (cosine motion)
                # duration calculation from env.goto_position
                distance = np.sum((traj[-1] - prev_angles)**2)**.5
                duration = distance / speed
                prev_angles = traj[-1]
                print(duration)
                duration = duration / len(traj) # equal duration (but not distance) for each waypoint

                for angles in traj:

                    # equal duration for each waypoint (linear motion)
                    # # duration calculation from env.goto_position
                    # distance = np.sum((angles - prev_angles)**2)**.5
                    # duration = distance / speed
                    # prev_angles = angles
                    # print(duration)

                    # next pypot trajectory waypoint
                    angle_dict = env.angle_dict(env.mirror_position(angles) if mirror else angles)
                    pypot_traj.append((duration, angle_dict))

                pypot_trajs.append(pypot_traj)

        with open('pypot_traj1.pkl', "wb") as f: pk.dump(pypot_trajs, f, protocol=2) 
   
    with open('traj1.pkl', "rb") as f: trajs = pk.load(f)

    # visualize traj
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)
    for _ in range(240*5): env.step(trajs[0][0][0])
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = -90,
        cameraPitch = 0,
        cameraTargetPosition = env.get_base()[0],
    )

    hang = True
    num_steps = 10

    input('.')

    for step in range(num_steps):

        if hang: input('.')
        for p,(traj, speed) in enumerate(trajs):
            print(p, traj[-1, [env.joint_index['l_ankle_y'], env.joint_index['l_knee_y']]])
            for t in range(len(traj)):
                env.goto_position(traj[t], speed=speed, duration=None, hang=(p==2))#False)
                if hang and p > 2: input(f"{p},{t}...")
    
            # pause between motions and make sure converge to last waypoint
            for _ in range(int(240*3.)): env.step(traj[-1])
            if hang: input(f"phase {['shift','push','kick','init'][p]} pause...")

        # mirror for next step
        trajs = [(
            np.stack([
                env.mirror_position(angles)
                for angles in traj]),
            speed)
            for (traj, speed) in trajs]

    final_angles = env.get_position()
    for _ in range(240*3): env.step(final_angles)
    if hang: input(f"done...")

