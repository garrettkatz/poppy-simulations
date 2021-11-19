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
from tabletop import add_table, add_cube

# integration APIs
import motor_control as mc
import vision as vz
import grasping as gr

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL)

    # this adds the table
    add_table()

    # add a few cubes at random positions on the table
    for c in range(3):
        pos = np.array([0, -.35, .41]) + (2*np.random.rand(3) - 1)*np.array([0.2, .02, 0])
        add_cube(tuple(pos), half_extents = (.01,)*3, rgb = tuple(np.random.rand(3)))

    # reposition robot to demonstrate walking
    pb.resetBasePositionAndOrientation(env.robot_id,
        posObj = (0, .4, .43),
        ornObj = pb.getQuaternionFromEuler((0, 0, 1)))

    input("..")

    # walking (Xulin, low priority)
    mc.walk_to(env,
        target_position = (0, 0, .43),
        target_orientation = pb.getQuaternionFromEuler((0, 0, 0)))

    # get head camera and arms into position
    angles = env.angle_dict(env.get_position())
    angles.update({"head_y": 80,
        "l_elbow_y": -145, "r_elbow_y": -145, 
        "l_shoulder_y": 45, "r_shoulder_y": 45, })
    env.goto_position(env.angle_array(angles), speed=2.)

    # input("..")

    # getting object positions/orientations from vision (Borui, low priority)
    objs = vz.get_object_poses(env)    
    for (pos, quat) in objs: print(pos, quat)

    # get all possible ways to grasp an object (Akshay, high priority)
    pos, quat = objs[0] # first cube for sake of example
    gripper_opening = .02 # open around cube
    targets = gr.get_possible_tip_targets_for_cube(pos, quat, gripper_opening)

    # tweak grasp points based on vision (Borui, high priority)
    for t in range(len(targets)):
        targets[t] = vz.tweak_grip(env, targets[t])

    # balanced reaching (Xulin, high priority)
    for t in range(len(targets)):

        # try IK on current target option
        joints = mc.balanced_reach_ik(env, targets[t], arm = "left")

        # stop when you find targets that work
        # for example, fingers don't collide (< +5+ degrees)
        gripper_angle = joints[env.joint_index["l_gripper"]]
        if gripper_angle < 5*np.pi/180: break

    env.goto_position(joints, 4, hang=False)

    # high-level manipulation planning

    for t in range(1000):
        pb.stepSimulation()

    input("..")

