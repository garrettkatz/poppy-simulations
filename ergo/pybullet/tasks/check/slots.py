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
def get_tip_targets(p, q, d):
    m = q
    t1 = p[0]-d*m[0], p[1]-d*m[3], p[2]-d*m[6]
    t2 = p[0]+d*m[0], p[1]+d*m[3], p[2]+d*m[6]
    return (t1, t2)
def get_tip_targets2(p, q, d):
    m = q
    t1 = p[0]-d*m[1], p[1]-d*m[4], p[2]-d*m[7]
    t2 = p[0]+d*m[1], p[1]+d*m[4], p[2]+d*m[7]
    return (t1, t2)
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
    slot2 = tt.add_box_compound(boxes)

    t_pos = tt.table_position()
    t_ext = tt.table_half_extents()
    s_pos = (t_pos[0], t_pos[1] + t_ext[1]/2, t_pos[2] + t_ext[2] + dims[2]/2)
    s_pos_2 = list(s_pos)
    s_pos_2[0] = s_pos_2[0]-0.1
    pb.resetBasePositionAndOrientation(slot, s_pos, (0.0, 0.0, 0.0, 1))
    pb.resetBasePositionAndOrientation(slot2, tuple(s_pos_2), (0.0, 0.0, 0.0, 1))


    d_pos = (s_pos[0], s_pos[1], s_pos[2] + dims[2]/2)
    d_ext = (dims[0]/4, dims[1]/4, dims[2]*0.75)
    disk = tt.add_cube(d_pos, d_ext, (0, 0, 1))

    angles = env.get_position()
   # input('.')
    count =0
    #while True:
    for i in range(10):

        #pb.stepSimulation()
        if i ==0:

            angles = env.angle_dict(env.get_position())

                # cube_pos = pb.getBasePositionAndOrientation(3)
            angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
            env.set_position(env.angle_array(angles))
        pos = pb.getBasePositionAndOrientation(disk)

        quat = pb.getMatrixFromQuaternion(pos[1])
        print(pos[0], pos[1])
        new_p = list(pos[0])
        new_p[2] = new_p[2] + 0.1
        new_o = pos[1]
        tar_args = get_tip_targets(new_p,quat,0.014)
        i_k = mc.balanced_reach_ik(env, tar_args, arm="right")
        env.goto_position(i_k,1)
        #for i in range(20):
            #pb.stepSimulation()
        print(pb.getBasePositionAndOrientation(3))

        print(pb.getBasePositionAndOrientation(4))
    #count = count+1

    new_p[2] = new_p[2] - 0.075
    new_o = pos[1]
    tar_args = get_tip_targets(new_p, quat, 0.014)
    i_k = mc.balanced_reach_ik(env, tar_args, arm="right")
    env.goto_position(i_k, 1)
    tar_args = get_tip_targets(new_p, quat, 0.002)
    i_k = mc.balanced_reach_ik(env, tar_args, arm="right")
    env.goto_position(i_k, 1)
    new_p[2] = new_p[2] + 0.2
    new_o = pos[1]
    tar_args = get_tip_targets(new_p, quat, 0.002)
    i_k = mc.balanced_reach_ik(env, tar_args, arm="right")
    env.goto_position(i_k, 1)

    tar_pos_orn = pb.getBasePositionAndOrientation(4)
    tar_pos = tar_pos_orn[0]
    d_pos = (tar_pos[0], tar_pos[1], new_p[2])
    tar_args = get_tip_targets(d_pos,quat,0.002)
    i_k = mc.balanced_reach_ik(env, tar_args, arm="right")
    env.goto_position(i_k, 1)


    input('.')