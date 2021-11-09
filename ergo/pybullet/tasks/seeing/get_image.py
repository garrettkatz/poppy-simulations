import os, sys
import numpy as np
import pybullet as pb

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
        pos = np.array([0, -.4, .41]) + np.random.randn(3)*np.array([0.2, .02, .01])
        add_cube(tuple(pos), half_extents = (.01,)*3, rgb = tuple(np.random.rand(3)))

    # angle head camera downward
    # put arms in field of view
    angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
    env.set_position(env.angle_array(angles))

    # gradually tilt head
    for t in range(200):

        angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35 + t/5})
        env.set_position(env.angle_array(angles))
    
        # this takes a picture
        rgba, view, proj = env.get_camera_image()

        # this steps the sim so blocks fall
        env.step()
    
    input("..")

