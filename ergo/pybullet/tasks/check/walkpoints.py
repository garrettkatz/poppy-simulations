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
    do_show = False
    num_waypoints = 20
    do_show = True
    num_waypoints = 3

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
    
        def iksolve(links, targets, angles, free, num_iters=1000, verbose=False):
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
    
            if verbose:
                env.set_position(angles)
                all_actual = get_pos()[1]
                all_actual = all_actual[all_links]
                print('iksolve errors:')
                print([env.joint_name[i] for i in range(env.num_joints) if i not in free])
                print(all_targets - all_actual)
                print(np.fabs(all_targets - all_actual).max())
    
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
                shift_angles = iksolve(links, targets, shift_angles, free, num_iters=2000)
    
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
            final_push_angles = iksolve(links, targets, push_angles, free, num_iters=2000)
    
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
                env.set_position(push_angles)
                _, push_jnt_pos, _ = settle(push_angles, zero_base, seconds=0)
    
                # set up back toe target
                links = [env.joint_index['l_toe']]
                targets = (push_jnt_pos[env.joint_index['r_toe']] - toe_to_toe)[np.newaxis]
                free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
    
                # re-solve knee/ankle to satisfy toe constraint
                push_angles = iksolve(links, targets, push_angles, free, num_iters=2000, verbose=False)
                env.set_position(push_angles)
    
                push_traj[t] = push_angles
                push_com_pos, push_jnt_pos, push_base = settle(push_angles, zero_base, seconds=1)
    
                print("PUSH", t)
            if show:
                show_support(push_com_pos, push_jnt_pos, ['r_toe', 'r_ankle_y', 'r_heel', 'l_toe', 'r_toe'])
                pt.show()
    
            ### SWING
    
            # set up heel target for swinging leg (reflect since back becomes front)
            links = [env.joint_index['l_heel']]
            targets = (push_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
            free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y', 'l_knee_y']]
            swing_angles = iksolve(links, targets, push_angles, free_joints, num_iters=3000, verbose=False)
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
                env.set_position(plant_angles)
                _, plant_jnt_pos, _ = settle(plant_angles, zero_base, seconds=0)

                # set up heel target for planted leg (reflect since back becomes front)
                links = [env.joint_index['l_heel']]
                targets = (swing_jnt_pos[env.joint_index['r_heel']] + reflectx(heel_to_heel))[np.newaxis]
                free_joints = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
                plant_angles = iksolve(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
                plant_com_pos, plant_jnt_pos, plant_base = settle(plant_angles, swing_base, seconds=1)
    
                # use knee position from interpolant, just re-solve ankle
                plant_angles = iksolve(links, targets, plant_angles, free_joints, num_iters=3000, verbose=False)
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
    
        # get waypoints
        trajs = get_traj(
            # angle from y-axis to front leg in initial stance
            init_front = .02*np.pi,
            # angle from back leg to y-axis in shift stance
            shift_back = .05*np.pi,
            # angle from y-axis to front leg in push stance
            push_front = -.05*np.pi,
            # angle from back leg to y-axis in push stance
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

