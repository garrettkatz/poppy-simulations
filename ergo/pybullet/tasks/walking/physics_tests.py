import numpy as np
import time
import pybullet as pb

import pybullet_data

client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0,0,-10)
plane_id = pb.loadURDF("plane.urdf")
robot_id = pb.loadURDF("test.urdf")#, basePosition=(0,0,2))

joint_idx = {}
for i in range(pb.getNumJoints(robot_id)):
    name = pb.getJointInfo(robot_id, i)[1].decode('utf-8')
    joint_idx[name] = i
print(joint_idx)

pb.resetBasePositionAndOrientation(robot_id, (0,0,2.5), (0,0,0,1))
pb.resetJointState(robot_id, joint_idx["left_hip"], -np.pi/6)
pb.resetJointState(robot_id, joint_idx["right_hip"], -np.pi/6)
pb.resetJointState(robot_id, joint_idx["left_ankle"], np.pi/6)
pb.resetJointState(robot_id, joint_idx["right_ankle"], np.pi/6)

for i in range(len(joint_idx)):
    pb.setJointMotorControl2(robot_id, i, pb.TORQUE_CONTROL, force=0.)

# input('.')
for _ in range(100):
    pb.stepSimulation()
    time.sleep(0.01)

input('.')

pb.enableJointForceTorqueSensor(robot_id, joint_idx["right_ankle"])

# pb.setJointMotorControl2(robot_id, joint_idx["left_ankle"], pb.POSITION_CONTROL, targetPosition=np.pi/8)
# while True:
for _ in range(100):
    pb.setJointMotorControl2(robot_id, joint_idx["right_ankle"], pb.TORQUE_CONTROL, force=0.)
    for i in range(len(joint_idx)):
        pb.setJointMotorControl2(robot_id, i, pb.TORQUE_CONTROL, force=0.)
    pb.stepSimulation()
    time.sleep(0.01)
    print(pb.getJointState(robot_id, joint_idx["right_ankle"]))
    # input('.')

pb.disconnect()

