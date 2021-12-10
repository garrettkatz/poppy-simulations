import os, sys
import numpy as np
import pybullet as pb

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
sys.path.append(os.path.join('..', 'walking'))
sys.path.append(os.path.join('..', 'seeing'))
sys.path.append(os.path.join('..', 'grasping'))

# common
from ergo import PoppyErgoEnv
import tabletop as tt

# integration APIs
import motor_control as mc
import vision as vz
import grasping as gr

def slot_arrays(inn, out):
    # inn/out: inner/outer half extents, as np arrays
    mid = (inn + out)/2
    ext = (out - inn)/2
    positions = np.array([
        [-mid[0], 0, 0],
        [+mid[0], 0, 0],
        [0, -mid[1], 0],
        [0, +mid[1], 0],
    ])
    extents = [
        (ext[0], out[1], out[2]),
        (ext[0], out[1], out[2]),
        (inn[0], ext[1], out[2]),
        (inn[0], ext[1], out[2]),
    ]
    
    return positions, extents

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)

    # this adds the table
    tt.add_table()

    # slot
    dims = np.array([.02, .04, .1])
    pos, ext = slot_arrays(dims/2, dims/2 + np.array([.01, .01, 0]))
    rgb = [(.75, .25, .25)]*4
    boxes = list(zip(map(tuple, pos), ext, rgb))

    # # define boxes that comprise slot structure
    # # (pos, ext, rgb)
    # boxes = [
    #     ( (-.1, 0, 0), (.02, .1, .1), (1, 0, 0) ),
    #     ( (+.1, 0, 0), (.02, .1, .1), (0, 1, 0) ),
    #     ( (0, -.1, 0), (.1, .02, .1), (1, 0, 0) ),
    #     ( (0, +.1, 0), (.1, .02, .1), (0, 1, 0) ),
    # ]
    
    slot = tt.add_box_compound(boxes)
    t_pos = tt.table_position()
    t_ext = tt.table_half_extents()
    s_pos = (t_pos[0], t_pos[1] + t_ext[1]/2, t_pos[2] + t_ext[2] + dims[2]/2)
    pb.resetBasePositionAndOrientation(slot, s_pos, (0.0, 0.0, 0.0, 1))

    d_pos = (s_pos[0], s_pos[1], s_pos[2] + dims[2]/2)
    d_ext = (dims[0]/2, dims[1]/2, dims[2])
    disk = tt.add_cube(d_pos, d_ext, (0, 0, 1))
    
    angles = env.get_position()
    input('.')
    while True:
        pb.stepSimulation()
        env.set_position(angles)

