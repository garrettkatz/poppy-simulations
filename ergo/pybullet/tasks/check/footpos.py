import os, sys
import numpy as np
import pybullet as pb

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)

    print(env.joint_name.values(), len(env.joint_name))

    # # toe
    # angles = np.zeros(env.num_joints)
    # angles[env.joint_index['r_hip_y']] = np.pi/8
    # # angles[env.joint_index['l_ankle_y']] = -np.pi/6
    # env.set_position(angles)
    # env.set_base(orn=pb.getQuaternionFromEuler((0.2*np.pi, 0, 0)))

    # heel
    angles = np.zeros(env.num_joints)
    angles[env.joint_index['r_hip_y']] = -np.pi/8
    # angles[env.joint_index['l_ankle_y']] = np.pi/6
    env.set_position(angles)
    env.set_base(orn=pb.getQuaternionFromEuler((-0.2*np.pi, 0, 0)))

    while True:
        env.step(action=angles)    

        pts = pb.getContactPoints(env.robot_id, linkIndexA = env.joint_index['l_ankle_y'])
        # for p,pt in enumerate(pts):
        #     posA, posB, normB, dist, force, = pt[5:10]
        #     print(f"pt {p}:")
        #     print(f"posA = {posA}")
        #     print(f"posB = {posB}")
        #     print(f"normB = {normB}")
        #     print(f"dist = {dist}")
        #     print(f"force = {force}")

        if len(pts) > 0:
            posA, posB, normB, dist, force, = pts[0][5:10]

            state = pb.getLinkState(env.robot_id, env.joint_index['l_ankle_y'])
            pos, orn = state[4:6] # joint

            M = pb.getMatrixFromQuaternion(orn) # orientation of ankle frame in world coordinates
            M = np.array(M).reshape(3,3)
            print(M)

            dx = np.array(posA) - np.array(pos)
            offset = np.dot(dx, M)
            # offset = np.dot(M, dx)
            print(offset)

            input('.')
    





