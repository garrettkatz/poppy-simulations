import pybullet as pb

def get_possible_tip_targets_for_cube(cube_pos, cube_quat, gripper_opening):
    # assumes the cube z-axis is aligned with the world z-axis

    p = cube_pos # cube center position
    m = pb.getMatrixFromQuaternion(cube_quat) # orientation matrix
    d = gripper_opening # distance between finger tips (0 is completely closed)

    # make list of all possible ways to grasp cube
    targets = []
    for a in [0, 1]: # grip along cube x-axis or y-axis
        for s in [-1, 1]: # thumb at positive or negative axis direction
            t1 = p[0] + s*d*m[a+0], p[1] + s*d*m[a+3], p[2] + s*d*m[a+6] # one fingertip
            t2 = p[0] - s*d*m[a+0], p[1] - s*d*m[a+3], p[2] - s*d*m[a+6] # other fingertip
            targets.append((t1, t2))

    return targets


