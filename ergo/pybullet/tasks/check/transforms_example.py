import os, sys
import numpy as np
import pybullet as pb

sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
sys.path.append(os.path.join('..', 'walking'))

from ergo import PoppyErgoEnv
import tabletop as tt
import motor_control as mc

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)
    angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90})
    env.set_position(env.angle_array(angles))

    final_block = tt.add_cube(
        position = np.array([-.1, -.3, .4]),
        half_extents = np.array([.01, .02, .1]),
        rgb = (0,1,0),
        mass = 0, # just for visual purposes, no physics
    )
    pb.resetBasePositionAndOrientation(final_block,
        posObj = (-.15, -.3, .5),
        ornObj = pb.getQuaternionFromEuler([.2, .2, .2]),
    )

    initial_block = tt.add_cube(
        position = np.array([-.1, -.3, .4]),
        half_extents = np.array([.01, .02, .1]),
        rgb = (1,0,0),
    )

    # disable collisions with final block
    pb.setCollisionFilterPair(final_block, initial_block, linkIndexA=-1, linkIndexB=-1, enableCollision=0)
    for link_index in range(env.num_joints):
        pb.setCollisionFilterPair(final_block, env.robot_id, linkIndexA=-1, linkIndexB=link_index, enableCollision=0)

    initial_joints = mc.balanced_reach_ik(env,
        tip_targets = ((-.092, -.3, .45), (-.108, -.3, .45)),
        arm="right")
    env.set_position(initial_joints)

    cam = (1.0, 0.7999724745750427, -23.40000343322754, (-0.023880355060100555, 0.42033448815345764, 0.27178314328193665))
    pb.resetDebugVisualizerCamera(*cam)

    # for t in range(10): env.step()

    env.step()
    input('.')

    # get transforms in world frame for initial/final block position and gripper
    init_world_pos, init_world_quat = pb.getBasePositionAndOrientation(initial_block)
    final_world_pos, final_world_quat = pb.getBasePositionAndOrientation(final_block)
    # final_world_pos, final_world_quat = (-.15, -.3, .5), pb.getQuaternionFromEuler([.2, .2, .2])

    grip_link_index = env.joint_index["r_gripper"]
    state = pb.getLinkState(env.robot_id, grip_link_index, computeForwardKinematics=1)
    grip_world_pos, grip_world_quat = state[:2] # COM
    # grip_world_pos, grip_world_quat = state[4:6] # link frame

    # get transform of gripper in initial block frame
    # init_world * grip_init = grip_world
    # grip_init = init_world**-1 * grip_world

    init_world_inv_pos, init_world_inv_quat = pb.invertTransform(
        init_world_pos, init_world_quat,
    )
    grip_init_pos, grip_init_quat = pb.multiplyTransforms(
        init_world_inv_pos, init_world_inv_quat,
        grip_world_pos, grip_world_quat,
    )

    # get transform of gripper target in world frame
    # targ_world = final_world * grip_init
    targ_world_pos, targ_world_quat = pb.multiplyTransforms(
        final_world_pos, final_world_quat,
        grip_init_pos, grip_init_quat,
    )

    # target visual check
    check_block = tt.add_cube(
        position = np.array([0,0,0]),
        half_extents = np.array([.05, .05, .05]),
        rgb = (0,0,1),
        mass = 0, # just for visual purposes, no physics
    )
    pb.resetBasePositionAndOrientation(check_block,
        posObj = targ_world_pos,
        ornObj = targ_world_quat,
    )
    input('.]')

    # # get transform in robot base frame for IK?
    # # base_world * targ_base = targ_world
    # # targ_base = base_world**-1 * targ_world
    # base_world_pos, base_world_quat = pb.getBasePositionAndOrientation(env.robot_id)
    # base_world_inv_pos, base_world_inv_quat = pb.invertTransform(
    #     base_world_pos, base_world_quat,
    # )
    # targ_base_pos, targ_base_quat = pb.multiplyTransforms(
    #     base_world_inv_pos, base_world_inv_quat,
    #     targ_world_pos, targ_world_quat,
    # )

    # do IK to gripper target
    angles = pb.calculateInverseKinematics(env.robot_id,
        grip_link_index,
        targ_world_pos,
        targ_world_quat,
        # targ_base_pos,
        # targ_base_quat,
        maxNumIterations=10000)
    a = 0
    target_joints = env.get_position()
    for r in range(env.num_joints):
        if not env.joint_fixed[r]:
            target_joints[r] = angles[a]
            a += 1

    env.goto_position(target_joints, speed=.5, hang=False)
    
    input('.')


