"""
To manually choose a new camera viewpoint and get its parameters:
1. Set MANUALLY_GET_NEW_CAM to True
2. Run this script, manually modify the view, then press Enter in the console
3. Copy-paste the camera parameters that get printed and use as args to resetDebugVisualizerCamera
"""
import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo import PoppyErgoEnv, convert_angles

# set this to True if you want to choose a new camera view
MANUALLY_GET_NEW_CAM = False

# set it to False to double-check the camera view that you copy-pasted

# this launches the simulator
env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)

# this loops until you get the camera view you want
while MANUALLY_GET_NEW_CAM:
    
    # Get the current camera view parameters
    cam = pb.getDebugVisualizerCamera()
    width, height, view, proj, camup, camfwd, horz, vert, yaw, pitch, dist, targ = cam
    # print(cam)
    # print(len(cam))

    # Render the camera image in the simulator
    _, _, rgb, depth, segment = pb.getCameraImage(width, height, view, proj)

    # Save the rendered camera image to file
    with open("getcam.pkl","wb") as f: pk.dump((rgb, depth, segment), f)

    # Print the parameters to be copy-pasted
    # can use these as input to pb.resetDebugVisualizerCamera()
    print("Camera parameters for resetDebugVisualizerCamera:")
    print(str((dist, yaw, pitch, targ)))

    # If dissatisfied, tweak the view and press Enter to print tweaked parameters
    input("ready...")

# Copy-paste the camera parameters here to check that it worked:
# cam = (1.2000000476837158, -2.4437904357910156e-06, -1.4000108242034912, (4.0046870708465576e-08, 0.8997313976287842, 0.37801095843315125))
cam = (1.0, 0.7999724745750427, -23.40000343322754, (-0.023880355060100555, 0.42033448815345764, 0.27178314328193665))

if not MANUALLY_GET_NEW_CAM:
    pb.resetDebugVisualizerCamera(*cam)

    # check inertia
    # confirms that pybullet reads mass from urdf, but recalculates inertia from collision shape
    info = pb.getDynamicsInfo(env.robot_id, env.joint_index["r_moving_tip"])
    mass, _, inertia = info[:3]
    print("mass", mass)
    print("inertia", inertia)

    input("start moving joints...")

    # set angles by name
    angles = env.angle_dict(env.get_position())
    for t in range(1, 6*180):
        angles.update({
            "r_wrist_y": -t / 12, "r_wrist_x":  t / 12, "r_gripper": -t / 12,
            "l_wrist_y": t / 12, "l_wrist_x":  t / 12, "l_gripper": -t / 12,
            "head_y": t / 12,
        })
    # for t in range(1, 2):
    #     angles.update({
    #         "r_wrist_y": 45., "r_wrist_x": 135., "r_gripper": 0.,
    #         "l_wrist_y": 45., "l_wrist_x": -135., "l_gripper": 0.,
    #         # "r_wrist_y": 0., "r_wrist_x": 180., "r_gripper": 0.,
    #         # "l_wrist_y": 0., "l_wrist_x": -180., "l_gripper": 0.,
    #         })
        env.set_position(env.angle_array(angles))
        env.step()
    

    input("waiting to close...")
