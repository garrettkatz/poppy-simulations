import pybullet as pb

def walk_to(env, target_position, target_orientation):
    pb.resetBasePositionAndOrientation(env.robot_id,
        posObj = target_position,
        ornObj = target_orientation)

def balanced_reach_ik(env, tip_targets, arm):
    neck_index = env.joint_index["head_z"]
    waist_index = 0
    link_indices = (
        env.joint_index[arm[0] + "_fixed_tip"], 
        env.joint_index[arm[0] + "_moving_tip"],
        neck_index,
        waist_index,
    )

    neck_position = pb.getLinkState(env.robot_id, neck_index)[0]
    waist_position = pb.getLinkState(env.robot_id, waist_index)[0]
    target_positions = tip_targets + (neck_position, waist_position)

    angles = env.inverse_kinematics(link_indices, target_positions, num_iters=2000)

    return angles
