import os, sys
import pickle as pk
import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

def reflectx(vec):
    return vec * np.array([-1, +1, +1]) # reflect vec through yz plane for when swing foot becomes support

def within_support(com_loc, jnt_loc, support_names):
    # names should be counter-clockwise vertices around support polygon
    urdfcom = com_loc.mean(axis=0)[:2]
    poly = np.array([jnt_loc[env.joint_index[name], :2] for name in support_names])
    for n in range(len(support_names)-1):
        uc = urdfcom - poly[n] # vertex to point
        uv = poly[n+1] - poly[n] # vertex to next vertex
        un = uv[[1, 0]] * np.array([-1, 1]) # edge normal
        if (uc*un).sum() > 0: return False
    return True

def get_init_waypoint(env, init_flat, init_abs_y):

    # initial waypoint pose
    init_angles = env.angle_array({
        'r_hip_y': -init_flat,
        'l_hip_y': +init_flat,
        'r_ankle_y': +init_flat,
        'l_ankle_y': -init_flat,
        'abs_y': init_abs_y,
    }, convert=False)
    _, jnt_loc = env.forward_kinematics(init_angles)

    init_oojl = False
    init_error = 0

    # translational offset from back to front toes/heels in init stance
    foot_to_foot = jnt_loc[env.joint_index['r_toe']] - jnt_loc[env.joint_index['l_toe']]

    return init_angles, init_oojl, init_error, foot_to_foot

def get_shift_waypoint(env, shift_swing, shift_torso, foot_to_foot):

    # shift waypoint pose
    shift_angles = env.angle_array({
        'r_hip_y': -shift_swing,
        'l_hip_y': +shift_swing,
        'r_ankle_y': +shift_swing,
        'l_ankle_y': -shift_swing,
        'abs_x': +shift_torso,
        'bust_x': -shift_torso,
        'l_shoulder_x': +shift_torso,
        'r_shoulder_x': -shift_torso,
    }, convert=False)
    _, jnt_loc = env.forward_kinematics(shift_angles)

    # set up front toe/heel target
    links = [env.joint_index[name] for name in ['r_toe','r_heel']]
    targets = np.stack((
        jnt_loc[env.joint_index['l_toe']] + foot_to_foot,
        jnt_loc[env.joint_index['l_heel']] + foot_to_foot,
    ))
    free = [env.joint_index[name] for name in ['r_knee_y', 'r_ankle_y']]
    shift_angles[env.joint_index['r_knee_y']] = 0.1 # to avoid out-of-joint-limit errors
    shift_angles, shift_oojl, shift_error = env.partial_ik(links, targets, shift_angles, free, num_iters=2000)

    return shift_angles, shift_oojl, shift_error

def get_push_waypoint(env, push_flat, push_swing, shift_torso, foot_to_foot):

    # push stance
    push_angles = env.angle_array({
        'r_hip_y': -push_flat,
        'l_hip_y': push_swing,
        'r_ankle_y': +push_flat,
        'abs_x': +shift_torso,
        'bust_x': -shift_torso,
        'l_shoulder_x': +shift_torso,
        'r_shoulder_x': -shift_torso,
    }, convert=False)
    _, jnt_loc = env.forward_kinematics(push_angles)

    # set up back toe target
    links = [env.joint_index['l_toe']]
    targets = (jnt_loc[env.joint_index['r_toe']] - foot_to_foot)[np.newaxis]
    free = [env.joint_index[name] for name in ['l_heel', 'l_ankle_y']]
    push_angles, push_oojl, push_error = env.partial_ik(links, targets, push_angles, free, num_iters=2000, verbose=False)

    return push_angles, push_oojl, push_error

def get_kick_waypoint(env, push_angles, foot_to_foot):

    _, jnt_loc = env.forward_kinematics(push_angles)

    # set up heel target for swinging leg (reflect since swing becomes flat)
    links = [env.joint_index['l_heel']]
    targets = (jnt_loc[env.joint_index['r_heel']] + reflectx(foot_to_foot))[np.newaxis]
    free = [env.joint_index[name] for name in ['l_toe', 'l_ankle_y']]
    kick_angles, kick_oojl, kick_error = env.partial_ik(links, targets, push_angles, free, num_iters=3000, verbose=False)

    return kick_angles, kick_oojl, kick_error

# right starts in front
def get_waypoints(env,
    # angle from vertical axis to flat (right) leg in initial stance
    init_flat,
    # angle for abs_y joint in initial stance
    init_abs_y,
    # angle from swing (left) leg to vertical axis in shift stance
    shift_swing,
    # angle of torso towards support leg in shift stance
    shift_torso,
    # angle from vertical axis to flat (right) leg in push stance
    push_flat,
    # angle from swing (left) leg to vertical axis in push stance
    push_swing,
):

    init_angles, init_oojl, init_error, foot_to_foot = get_init_waypoint(env, init_flat, init_abs_y)
    shift_angles, shift_oojl, shift_error = get_shift_waypoint(env, shift_swing, shift_torso, foot_to_foot)
    push_angles, push_oojl, push_error = get_push_waypoint(env, push_flat, push_swing, shift_torso, foot_to_foot)
    kick_angles, kick_oojl, kick_error = get_kick_waypoint(env, push_angles, foot_to_foot)

    return (
        (init_angles, init_oojl, init_error),
        (shift_angles, shift_oojl, shift_error),
        (push_angles, push_oojl, push_error),
        (kick_angles, kick_oojl, kick_error),
    )

if __name__ == "__main__":

    do_show = False
    do_gait_param_fig = False
    do_phase_waypoint_fig = True

    pt.rcParams["text.usetex"] = True
    pt.rcParams['font.family'] = 'serif'

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=do_show)
    base = env.get_base()
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1,
        cameraYaw = -90,
        cameraPitch = 0,
        cameraTargetPosition = base[0],
    )

    jnt_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel", "ankle_y", "knee_y")]
    ft_idx = [env.joint_index[f"{lr}_{jnt}"] for lr in "lr" for jnt in ("toe", "heel")]
    xwid, yh = .3, .5

    if do_gait_param_fig:

        init_angles, _, _, f2f, = get_init_waypoint(env, init_flat = np.pi/6, init_abs_y=np.pi/4)
        _, init_jnt_loc = env.forward_kinematics(init_angles)

        shift_angles, shift_oojl, shift_error = get_shift_waypoint(env, shift_swing=np.pi/4, shift_torso=np.pi/4, foot_to_foot=f2f)
        _, shift_jnt_loc = env.forward_kinematics(shift_angles)

        pt.figure(figsize=(4, 2))
        pt.subplot(1,2,1)
        
        for j in jnt_idx:
            p = env.joint_parent[j]
            if p == -1: continue
            color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
            pt.plot(-init_jnt_loc[[p,j],1], init_jnt_loc[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
        for lr in "lr":
            color, zorder = (.5, 0) if lr == "l" else (0, 1)
            j, p = env.joint_index[f"{lr}_toe"], env.joint_index[f"{lr}_heel"]
            pt.plot(-init_jnt_loc[[p,j],1], init_jnt_loc[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)

        hip = init_jnt_loc[env.joint_index['r_hip_y']]
        pt.plot([-hip[1], .1-hip[1]], [hip[2], .1+hip[2]], 'k.-')
        pt.plot([-hip[1], -hip[1]], [.2+hip[2], 0], 'k:')
        pt.text(-.07, .2, '$\\theta^{\\phi}_{s}$')
        pt.text(+.03, .2, '$\\theta^{\\phi}_{f}$')
        pt.text(+.03, .5, '$\\theta^{\\phi}_{y}$')
        pt.axis('off')
        pt.axis('equal')

        pt.subplot(1,2,2)
        # compensate for urdf leg joint frame offsets
        for lr in "lr":
            for name in ('ankle_y', 'knee_y'):
                shift_jnt_loc[env.joint_index[f"{lr}_{name}"], 0] = shift_jnt_loc[env.joint_index[f"{lr}_heel"], 0]
        for j in range(len(shift_jnt_loc)):
            # skip gripper, head cam
            if env.joint_name[j][2:] in ('gripper', 'wrist_x', 'fixed_tip', 'moving_tip'): continue
            if env.joint_name[j] == 'head_cam': continue
            color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
            p = env.joint_parent[j]
            if p == -1:
                pt.plot([base[0][0], shift_jnt_loc[j,0]], [base[0][2], shift_jnt_loc[j,2]], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
            else:
                pt.plot(shift_jnt_loc[[p,j],0], shift_jnt_loc[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)

        pt.plot([base[0][0], base[0][0]], [base[0][2], .2+base[0][2]], 'k:')
        pt.text(-.065, .56, '$\\theta^{\\phi}_{x}$')
        pt.axis('off')
        pt.axis('equal')

        pt.tight_layout()
        pt.savefig('params.eps')
        pt.show()

    if do_phase_waypoint_fig:

        # gait waypoint fig
        # waypoints = get_waypoints(env,
        #     # angle from vertical axis to flat leg in initial stance
        #     init_flat = 0.03141593,#.15*np.pi,
        #     # angle for abs_y joint in initial stance
        #     init_abs_y = np.pi/16,
        #     # angle from back leg to vertical axis in shift stance
        #     shift_swing = 0.13912767,#.05*np.pi,
        #     # angle of torso towards support leg in shift stance
        #     shift_torso = np.pi/5.75,
        #     # angle from vertical axis to flat (right) leg in push stance
        #     push_flat = 0.0448799,#-.05*np.pi,
        #     # angle from back (left) leg to vertical axis in push stance
        #     push_swing = -0.15707963,#-.01*np.pi,
        # )
        # orig fig
        waypoints = get_waypoints(env,
            # angle from vertical axis to flat leg in initial stance
            init_flat = .02*np.pi,
            # angle for abs_y joint in initial stance
            init_abs_y = np.pi/16,
            # angle from swing leg to vertical axis in shift stance
            shift_swing = .05*np.pi,
            # angle of torso towards support leg in shift stance
            shift_torso = np.pi/5.75,
            # angle from vertical axis to flat leg in push stance
            push_flat = -.02*np.pi,#-.05*np.pi,
            # angle from swing leg to vertical axis in push stance
            push_swing = -.08*np.pi,#-.01*np.pi,
        )

        pt.figure(figsize=(8, 4))
        for w, (angles, oojr, error) in enumerate(waypoints):
            print(angles[env.joint_index['l_ankle_y']])

            com_loc, jnt_loc = env.forward_kinematics(angles)

            pt.subplot(2, 4, w+1)
            for j in jnt_idx:
                p = env.joint_parent[j]
                if p == -1: continue
                color, zorder = (.5, 0) if env.joint_name[j][:2] == "l_" else (0, 1)
                pt.plot(-jnt_loc[[p,j],1], jnt_loc[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)
            for lr in "lr":
                color, zorder = (.5, 0) if lr == "l" else (0, 1)
                j, p = env.joint_index[f"{lr}_toe"], env.joint_index[f"{lr}_heel"]
                pt.plot(-jnt_loc[[p,j],1], jnt_loc[[p,j],2], marker='.', linestyle='-', color=(color,)*3, zorder=zorder)

            foot = jnt_loc[ft_idx].mean(axis=0)
            pt.ylim([foot[2]-.05, foot[2]+yh])
            pt.xlim([foot[1]-xwid, foot[1]+xwid])
            pt.title(('Initial', 'Shift', 'Push', 'Kick')[w])
            pt.axis('off')
            # pt.axis('equal')

            pt.subplot(2, 4, 4+w+1)
            CoM = com_loc.mean(axis=0)
            names = ('r_toe', 'r_heel', 'l_heel', 'l_toe', 'r_toe')
            if w == 2: names = ('r_toe', 'r_heel', 'l_toe', 'r_toe')
            if w == 3: names = ('r_toe', 'r_heel', 'l_heel', 'r_toe')
            support_polygon = np.array([jnt_loc[env.joint_index[name]] for name in names])
            pt.plot(support_polygon[:,0], support_polygon[:,1], 'k.-')
            pt.plot(CoM[0], CoM[1], 'ko')
            pt.text(CoM[0]+.007, CoM[1], 'CoM')
            for name in names[:-1]:
                idx = env.joint_index[name]
                pt.text(jnt_loc[idx,0]+.015, jnt_loc[idx,1]-.01, name)
            toe = jnt_loc[env.joint_index['r_toe']]
            pt.xlim([toe[0]-.01, toe[0]+.15])
            pt.ylim([toe[1]-.01, toe[1]+.2])
            pt.title(f"CoM {within_support(com_loc, jnt_loc, names)}, {error:.4f}")
            pt.axis('off')
            # pt.axis('equal')

        pt.tight_layout()
        pt.savefig('waypoints.eps')
        pt.show()
