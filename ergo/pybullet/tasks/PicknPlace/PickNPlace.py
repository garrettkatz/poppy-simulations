import os, sys
import numpy as np
import pybullet as pb
def get_tip_targets(p, q, d):
    m = q
    t1 = p[0]-d*m[0], p[1]-d*m[3], p[2]-d*m[6]
    t2 = p[0]+d*m[0], p[1]+d*m[3], p[2]+d*m[6]
    return (t2, t1)
if __name__ == "__main__":
    sys.path.append(os.path.join('..', '..', 'envs'))
    from ergo import PoppyErgoEnv

    sys.path.append(os.path.join('..', '..', 'objects'))
    from tabletop import add_table, add_cube

    # this launches the simulator
    # fix the base to avoid balance issues
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)

    # this adds the table
    add_table()

    # add a few cubes at random positions on the table
for c in range(5):
    pos = np.array([0, -.4, .41]) + np.random.randn(3) * np.array([0.2, .02, .01])
    add_cube(tuple(pos), half_extents=(.01,) * 3, rgb=tuple(np.random.rand(3)))

    # angle head camera downward
    # put arms in field of view

angles = env.angle_dict(env.get_position())
#cube_pos = pb.getBasePositionAndOrientation(3)
angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
env.set_position(env.angle_array(angles))
pos = pb.getBasePositionAndOrientation(3)
quat = pb.getMatrixFromQuaternion(pos[1])
print(pos[0],pos[1])
new_p = pos[0]
newP_list = list(new_p)
newP_list[1]=newP_list[1]+0.1
new_p_tuple = tuple(newP_list)

#position close to block
tar_args = get_tip_targets(new_p_tuple,quat,0.02)
joints_new = env.inverse_kinematics([35,34],tar_args)
env.goto_position(joints_new)#movement

#position set on block
tar_args = get_tip_targets(new_p,quat,0.02)
i_k = env.inverse_kinematics([35, 34],tar_args)
env.goto_position(i_k)#movement

#close gripper
tar_args = get_tip_targets(new_p,quat,0.008)
i_k = env.inverse_kinematics([35, 34],tar_args)
env.goto_position(i_k) #movement

block_two_position = pb.getBasePositionAndOrientation(4)
quat_two = pb.getMatrixFromQuaternion(block_two_position[1])
block_two_new_pos = list(block_two_position[0])
block_two_new_pos[2] = block_two_new_pos[2]+0.05
tuple_pos_two = tuple(block_two_new_pos)
tar_args = get_tip_targets(tuple_pos_two,quat_two,0.008)
i_k = env.inverse_kinematics([35, 34],tar_args)
env.goto_position(i_k) #movement
print(angles)
    # gradually tilt head
'''  for t in range(200):
        angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35 + t / 5})
        env.set_position(env.angle_array(angles))

        # this takes a picture
        rgba, view, proj = env.get_camera_image()

        # this steps the sim so blocks fall
        env.step()
'''
input("..")