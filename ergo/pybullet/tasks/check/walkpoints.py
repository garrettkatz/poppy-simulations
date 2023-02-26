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
    show_filter = True
    do_show = False
    num_waypoints = 20
    # do_show = True
    # num_waypoints = 3

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

        def within_support(com_pos, jnt_pos, names):
            mx64com = jnt_pos[MX64IDX].mean(axis=0)
            mx28com = jnt_pos[MX28IDX].mean(axis=0)
            urdfcom = com_pos.mean(axis=0)[:2]
            poly = np.array([jnt_pos[env.joint_index[name], :2] for name in names])
            for n in range(len(names)-1):
                uc = urdfcom - poly[n] # vertex to point
                uv = poly[n+1] - poly[n] # vertex to next vertex
                un = uv[[1, 0]] * np.array([-1, 1]) # edge normal
                if (uc*un).sum() > 0: return False
            return True
    
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
                    if verbose:
                        print(f"{env.joint_name[i]}: {angles[i]} not in [{env.joint_low[i]}, {env.joint_high[i]}]!")
                    oojr = True
            if oojr and verbose: input('uh oh...')

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
    
        def reflectx(vec):
            return vec * np.array([-1, +1, +1]) # reflect through yz plane for when back foot becomes front
    
        def settle(angles, base=None, seconds=1):
            if base is not None: env.set_base(*base)
            env.set_position(angles)
            for _ in range(int(240*seconds)): env.step(angles)
            com_pos, jnt_pos = get_pos()
            base = env.get_base()
            return com_pos, jnt_pos, base

        def get_init_waypoint(init_front, init_abs_y, zero_base):

            # initial waypoint pose
            init_angles = env.angle_array({
                'r_hip_y': -init_front,
                'l_hip_y': +init_front,
                'r_ankle_y': +init_front,
                'l_ankle_y': -init_front,
                'abs_y': init_abs_y,
            }, convert=False)
            _, init_jnt_pos, _ = settle(init_angles, base=zero_base, seconds=0)

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
            _, shift_jnt_pos, _ = settle(shift_angles, base=zero_base, seconds=0)

            # set up front toe/heel target
            links = [env.joint_index[name] for name in ['r_toe','r_heel']]
            targets = np.stack((
                shift_jnt_pos[env.joint_index['l_toe']] + foot_to_foot,
                shift_jnt_pos[env.joint_index['l_heel']] + foot_to_foot,
            ))
            free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
            shift_angles, shift_oojr, shift_error = iksolve(links, targets, shift_angles, free, num_iters=2000)

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
            _, push_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)
    
            # set up back toe target
            links = [env.joint_index['l_toe']]
            targets = (push_jnt_pos[env.joint_index['r_toe']] - foot_to_foot)[np.newaxis]
            free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
            push_angles, push_oojr, push_error = iksolve(links, targets, push_angles, free, num_iters=2000)
            push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base, seconds=0)

            return push_angles, push_oojr, push_error

        def get_kick_waypoint(push_angles, zero_base, foot_to_foot):

            _, push_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)

            # set up heel target for swinging leg (reflect since back becomes front)
            links = [env.joint_index['l_heel']]
            targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(foot_to_foot))[np.newaxis]
            free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y', 'l_knee_y']]
            kick_angles, kick_oojr, kick_error = iksolve(links, targets, push_angles, free_joints, num_iters=3000, verbose=False)

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
            _, _, zero_base = settle(zero_angles)

            init_angles, init_oojr, init_error, foot_to_foot = get_init_waypoint(init_front, init_abs_y, zero_base)
            shift_angles, shift_oojr, shift_error = get_shift_waypoint(shift_back, shift_torso, zero_base, foot_to_foot)
            push_angles, push_oojr, push_error = get_push_waypoint(push_front, push_back, shift_torso, zero_base, foot_to_foot)
            kick_angles, kick_oojr, kick_error = get_kick_waypoint(push_angles, zero_base, foot_to_foot)

            return (
                (init_angles, init_oojr, init_error),
                (shift_angles, shift_oojr, shift_error),
                (push_angles, push_oojr, push_error),
                (kick_angles, kick_oojr, kick_error),
            )

        def interpolate_waypoints():
            pass
    
        def get_traj(
            # angle from vertical axis to front leg in initial stance
            init_front,
            # angle from back leg to vertical axis in shift stance
            shift_back,
            # angle from vertical axis to front leg in push stance
            push_front,
            # angle from back leg to vertical axis in push stance
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
            init_abs_y = np.pi/16
        
            init_angles = np.zeros(env.num_joints)
            init_angles[env.joint_index['r_hip_y']] = -init_front
            init_angles[env.joint_index['l_hip_y']] = +init_front
            init_angles[env.joint_index['r_ankle_y']] = +init_front
            init_angles[env.joint_index['l_ankle_y']] = -init_front
            # init_angles[env.joint_index['r_shoulder_x']] = -np.pi/8
            # init_angles[env.joint_index['l_shoulder_x']] = +np.pi/8
            init_angles[env.joint_index['abs_y']] = init_abs_y
            init_com_pos, init_jnt_pos, init_base = settle(init_angles, base=zero_base, seconds=2)
        
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
            shift_torso = np.pi/5.75
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

                shift_com_pos, shift_jnt_pos, shift_base = settle(shift_angles, zero_base, seconds=1)
    
                # initialize positive knee angle to avoid out-of-joint-range solutions
                shift_angles[env.joint_index['r_knee_y']] = 0.1
    
                # set up front toe/heel target
                links = [env.joint_index[name] for name in ['r_toe','r_heel']]
                targets = np.stack((
                    shift_jnt_pos[env.joint_index['l_toe']] + toe_to_toe,
                    shift_jnt_pos[env.joint_index['l_heel']] + heel_to_heel,
                ))
                free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
                shift_angles, _, _ = iksolve(links, targets, shift_angles, free, num_iters=2000)
    
                env.set_position(shift_angles)
    
                shift_traj[t] = shift_angles
                shift_com_pos, shift_jnt_pos, shift_base = settle(shift_angles, zero_base, seconds=1)
    
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

            push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base)
    
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
            final_push_angles, _, _ = iksolve(links, targets, push_angles, free, num_iters=2000)
    
            push_com_pos, push_jnt_pos, push_base = settle(final_push_angles, zero_base)#, seconds=0.1)
            print("PUSH final")
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
            push_traj = np.empty((num_waypoints, env.num_joints))
            # for t,w in enumerate(np.linspace(0, 1, num_waypoints)):
            for t,w in enumerate(.5 - .5*np.cos(np.linspace(0, np.pi, num_waypoints))):
    
                # get link positions at interpolated waypoint
                push_angles = w*final_push_angles + (1-w)*final_shift_angles
                _, push_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)
    
                # set up back toe target
                links = [env.joint_index['l_toe']]
                targets = (push_jnt_pos[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
    
                # re-solve knee/ankle to satisfy toe constraint
                push_angles, _, _ = iksolve(links, targets, push_angles, free, num_iters=2000, verbose=False)
                env.set_position(push_angles)
    
                push_traj[t] = push_angles
                push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base, seconds=1)
    
                print("PUSH", t)
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
            ### KICK
    
            # set up heel target for swinging leg (reflect since back becomes front)
            links = [env.joint_index['l_heel']]
            targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
            free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y', 'l_knee_y']]
            swing_angles, _, _ = iksolve(links, targets, push_angles, free_joints, num_iters=3000, verbose=False)
            swing_com_pos, swing_jnt_pos, swing_base = settle(swing_angles, push_base, seconds=3)
    
            # fast swing
            swing_traj = swing_angles[np.newaxis,:]
    
            print("SWING")
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
                _, plant_jnt_pos, _ = settle(plant_angles, zero_base, seconds=0)

                # set up heel target for planted leg (reflect since back becomes front)
                links = [env.joint_index['l_heel']]
                targets = (swing_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
                free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
                plant_angles, _, _ = iksolve(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
                plant_com_pos, plant_jnt_pos, plant_base = settle(plant_angles, swing_base, seconds=1)
    
                # use knee position from interpolant, just re-solve ankle
                plant_angles, _, _ = iksolve(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
                env.set_position(plant_angles)
    
                plant_traj[t] = plant_angles
                plant_com_pos, plant_jnt_pos, plant_base = settle(plant_angles, zero_base, seconds=1)
    
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

        if do_gait_figs:

            jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]
            ft_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel")]
            xwid, yh = .3, .5
    
            waypoints = get_waypoints(
                # angle from vertical axis to front leg in initial stance
                init_front = .15*np.pi,
                # angle for abs_y joint in initial stance
                init_abs_y = np.pi/16,
                # angle from back leg to vertical axis in shift stance
                shift_back = .05*np.pi,
                # angle of torso towards support leg in shift stance
                shift_torso = np.pi/5.75,
                # angle from vertical axis to front leg in push stance
                push_front = -.05*np.pi,
                # angle from back leg to vertical axis in push stance
                push_back = -.01*np.pi,
                # whether to visualize
                show = do_show,
            )
    
            pt.figure(figsize=(4, 3))
            pt.subplot(1,2,1)
            
            angles, _, _ = waypoints[0]
            com_pos, jnt_pos, base = settle(angles, seconds=0)
    
            for j in jnt_idx:
                p = env.joint_parent[j]
                if p == -1: continue
                color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
                pt.plot(-jnt_pos[[p,j],1], jnt_pos[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
            for lr in "lr":
                color, zorder = (.5, 0) if lr == "l" else (0, 1)
                j, p = env.joint_index[f"{lr}_toe"], env.joint_index[f"{lr}_heel"]
                pt.plot(-jnt_pos[[p,j],1], jnt_pos[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
    
            hip = jnt_pos[env.joint_index['r_hip_y']]
            pt.plot([-hip[1], .1-hip[1]], [hip[2], .1+hip[2]], 'k.-')
            pt.plot([-hip[1], -hip[1]], [.2+hip[2], 0], 'k:')
            pt.text(-.05, .2, '$\\theta^{\\phi}_{\\ell}$')
            pt.text(+.03, .2, '$\\theta^{\\phi}_{r}$')
            pt.text(+.03, .5, '$\\theta^{\\phi}_{y}$')
            pt.axis('off')
            pt.axis('equal')

            for w,(angles, _, _) in enumerate(waypoints):
                com_pos, jnt_pos, base = settle(angles, seconds=0)
                names = ('r_toe', 'r_heel', 'l_heel', 'l_toe', 'r_toe')
                if w in [1,2]: names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
                if w == 3: names = ('r_toe', 'r_heel', 'l_heel', 'r_toe')
                print(within_support(com_pos, jnt_pos, names))

            waypoints = get_waypoints(
                # angle from vertical axis to front leg in initial stance
                init_front = .02*np.pi,
                # angle for abs_y joint in initial stance
                init_abs_y = np.pi/16,
                # angle from back leg to vertical axis in shift stance
                shift_back = .05*np.pi,
                # angle of torso towards support leg in shift stance
                shift_torso = np.pi/5.75,
                # angle from vertical axis to front leg in push stance
                push_front = -.05*np.pi,
                # angle from back leg to vertical axis in push stance
                push_back = -.01*np.pi,
                # whether to visualize
                show = do_show,
            )
    
            pt.subplot(1,2,2)
            angles, _, _ = waypoints[1]
            env.set_position(angles)
            com_pos, jnt_pos, base = settle(angles, seconds=0)
            # compensate for urdf leg joint frame offsets
            for lr in "lr":
                for name in ('ankle_y', 'knee_y'):
                    jnt_pos[env.joint_index[f"{lr}_{name}"], 0] = jnt_pos[env.joint_index[f"{lr}_heel"], 0]
            for j in range(len(jnt_pos)):
                # skip gripper, head cam
                if env.joint_name[j][2:] in ('gripper', 'wrist_x', 'fixed_tip', 'moving_tip'): continue
                if env.joint_name[j] == 'head_cam': continue
                color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
                p = env.joint_parent[j]
                if p == -1:
                    pt.plot([base[0][0], jnt_pos[j,0]], [base[0][2], jnt_pos[j,2]], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
                else:
                    pt.plot(jnt_pos[[p,j],0], jnt_pos[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
    
            pt.plot([base[0][0], base[0][0]], [base[0][2], .2+base[0][2]], 'k:')
            pt.text(-.05, .56, '$\\theta^{\\phi}_{x}$')
            pt.axis('off')
            pt.axis('equal')
    
            pt.tight_layout()
            pt.savefig('params.eps')
            pt.show()
    
            pt.figure(figsize=(8, 4))
            for w, (angles, oojr, error) in enumerate(waypoints):
    
                env.set_position(angles)
                com_pos, jnt_pos, base = settle(angles, seconds=0)
    
                pt.subplot(2, 4, w+1)
                # for j in range(len(jnt_pos)):
                for j in jnt_idx:
                    p = env.joint_parent[j]
                    if p == -1: continue
                    color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
                    pt.plot(-jnt_pos[[p,j],1], jnt_pos[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
                for lr in "lr":
                    color, zorder = (.5, 0) if lr == "l" else (0, 1)
                    j, p = env.joint_index[f"{lr}_toe"], env.joint_index[f"{lr}_heel"]
                    pt.plot(-jnt_pos[[p,j],1], jnt_pos[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
    
                foot = jnt_pos[ft_idx].mean(axis=0)
                pt.ylim([foot[2]-.05, foot[2]+yh])
                pt.xlim([foot[1]-xwid, foot[1]+xwid])
                pt.title(('Initial', 'Shift', 'Push', 'Kick')[w])
                pt.axis('off')
                # pt.axis('equal')
    
                pt.subplot(2, 4, 4+w+1)
                CoM = com_pos.mean(axis=0)
                names = ('r_toe', 'r_heel', 'l_heel', 'l_toe', 'r_toe')
                if w == 2: names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
                if w == 3: names = ('r_toe', 'r_heel', 'l_heel', 'r_toe')
                support_polygon = np.array([jnt_pos[env.joint_index[name]] for name in names])
                pt.plot(support_polygon[:,0], support_polygon[:,1], 'k.-')
                pt.plot(CoM[0], CoM[1], 'ko')
                pt.text(CoM[0]+.007, CoM[1], 'CoM')
                for name in names[:-1]:
                    idx = env.joint_index[name]
                    pt.text(jnt_pos[idx,0]+.015, jnt_pos[idx,1]-.01, name)
                toe = jnt_pos[env.joint_index['r_toe']]
                pt.xlim([toe[0]-.01, toe[0]+.15])
                pt.ylim([toe[1]-.01, toe[1]+.2])
                # pt.title(within_support(com_pos, jnt_pos, names))
                pt.axis('off')
                # pt.axis('equal')
    
            pt.tight_layout()
            pt.savefig('waypoints.eps')
            pt.show()

            env.close()
            import sys
            sys.exit()

        ### do IK and CoM filter
        if do_filter or show_filter:

            grid_sampling = 8
            init_front_range = np.linspace(0.01*np.pi, np.pi/6, grid_sampling)
            init_abs_y = np.pi/16
            shift_back_range = np.linspace(0.01*np.pi, np.pi/4, grid_sampling)
            shift_torso = np.pi/5.75
            push_front_range = np.linspace(-.1*np.pi, 0.05*np.pi, grid_sampling)
            push_back_range = np.linspace(-np.pi/6, 0, grid_sampling)

        if do_filter:

            # get base in zero joint angle pose
            zero_angles = np.zeros(env.num_joints)
            _, _, zero_base = settle(zero_angles)
    
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
                    com_pos, jnt_pos, _ = settle(shift_angles, zero_base, seconds=0)
                    names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
                    shift_support[i,j] = within_support(com_pos, jnt_pos, names)

            swing_oojr = np.zeros((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            swing_error = -np.ones((len(foot_to_foot), len(push_front_range), len(push_back_range)))
            swing_support = np.zeros((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            # swing_clear = np.zeros((len(foot_to_foot), len(push_front_range), len(push_back_range)), dtype=bool)
            for i, init_front in enumerate(init_front_range):
                for j, push_front in enumerate(push_front_range):
                    for k, push_back in enumerate(push_back_range):
                        print(f"shift {i},{j},{k}/{len(init_front_range)},{len(push_front_range)},{len(push_back_range)}")
                        push_angles, push_oojr, push_error = get_push_waypoint(push_front, push_back, shift_torso, zero_base, foot_to_foot[i])
                        kick_angles, kick_oojr, kick_error = get_kick_waypoint(push_angles, zero_base, foot_to_foot[i])
                        swing_oojr[i,j,k] = push_oojr or kick_oojr
                        swing_error[i,j,k] = max(push_error, kick_error)

                        com_pos, jnt_pos, _ = settle(push_angles, zero_base, seconds=0)
                        push_support = within_support(com_pos, jnt_pos, ('r_toe', 'r_heel', 'l_toe', 'r_toe'))
                        com_pos, jnt_pos, _ = settle(kick_angles, zero_base, seconds=0)
                        kick_support = within_support(com_pos, jnt_pos, ('r_toe', 'r_heel', 'l_heel', 'r_toe'))
                        swing_support[i,j,k] = push_support and kick_support
    
                        # # check clearance
                        # _, push_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)
                        # _, kick_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)

            with open('filter.pkl','wb') as f:
                pk.dump((shift_oojr, shift_error, shift_support, swing_oojr, swing_error, swing_support), f)

        if show_filter:

            with open('filter.pkl','rb') as f:
                (shift_oojr, shift_error, shift_support, swing_oojr, swing_error, swing_support) = pk.load(f)

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
            pt.savefig('shift_search.eps')
            pt.show()

            fig, ax = pt.subplots(2, len(init_front_range), figsize=(13,3.5), constrained_layout=True)
            for i, init_front in enumerate(init_front_range):
                ax[0,i].imshow(~swing_oojr[i] & swing_support[i], cmap='gray')
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
            pt.savefig('swing_search.eps')
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
        trajs = get_traj(
            # angle from vertical axis to front leg in initial stance
            init_front = .02*np.pi,
            # angle from back leg to vertical axis in shift stance
            shift_back = .05*np.pi,
            # angle from vertical axis to front leg in push stance
            push_front = -.05*np.pi,
            # angle from back leg to vertical axis in push stance
            push_back = -.01*np.pi,
            num_waypoints = num_waypoints,
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
    for _ in range(240*1): env.step(trajs[0][0][0])
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = -90,
        cameraPitch = 0,
        cameraTargetPosition = env.get_base()[0],
    )

    hang = False
    num_steps = 10

    input('.')

    for step in range(num_steps):

        if hang: input('.')
        for (traj, speed) in trajs:
            for t in range(len(traj)):
                env.goto_position(traj[t], speed=speed, duration=None, hang=False)
                if hang: input(f"{t}...")
    
            # pause between motions and make sure converge to last waypoint
            for _ in range(int(240*1.)): env.step(traj[-1])
            if hang: input(f"pause...")

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

