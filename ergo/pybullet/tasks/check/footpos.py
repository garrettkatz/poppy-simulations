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

    angles = np.zeros(env.num_joints)
    # toe
    angles[env.joint_index['r_hip_y']] = -np.pi/2
    angles[env.joint_index['l_hip_y']] = -np.pi/2
    env.set_base(orn=pb.getQuaternionFromEuler((3*np.pi/4, 0, 0)))
    # heel
    angles[env.joint_index['r_hip_y']] = -np.pi/2
    angles[env.joint_index['l_hip_y']] = -np.pi/2
    env.set_base(orn=pb.getQuaternionFromEuler((1*np.pi/4, 0, 0)))

    env.set_position(angles)
    input('.')


    # # toe
    # angles = np.zeros(env.num_joints)
    # angles[env.joint_index['r_hip_y']] = np.pi/8
    # # angles[env.joint_index['l_ankle_y']] = -np.pi/6
    # env.set_position(angles)
    # env.set_base(orn=pb.getQuaternionFromEuler((0.2*np.pi, 0, 0)))

    # # heel
    # angles = np.zeros(env.num_joints)
    # angles[env.joint_index['r_hip_y']] = -np.pi/8
    # # angles[env.joint_index['l_ankle_y']] = np.pi/6
    # env.set_position(angles)
    # env.set_base(orn=pb.getQuaternionFromEuler((-0.2*np.pi, 0, 0)))

    while True:
        env.step(action=angles)    

        left_pts = pb.getContactPoints(env.robot_id, linkIndexA = env.joint_index['l_ankle_y'])
        right_pts = pb.getContactPoints(env.robot_id, linkIndexA = env.joint_index['r_ankle_y'])
        # for p,pt in enumerate(pts):
        #     posA, posB, normB, dist, force, = pt[5:10]
        #     print(f"pt {p}:")
        #     print(f"posA = {posA}")
        #     print(f"posB = {posB}")
        #     print(f"normB = {normB}")
        #     print(f"dist = {dist}")
        #     print(f"force = {force}")

        print(len(left_pts), len(right_pts))
        # input('.')

        pts = left_pts
        if len(left_pts) == len(right_pts) == 1:
            lposA, posB, normB, dist, force, = left_pts[0][5:10]
            rposA, posB, normB, dist, force, = right_pts[0][5:10]
        else:
            continue

        state = pb.getLinkState(env.robot_id, env.joint_index['l_ankle_y'])
        lpos, lorn = state[4:6] # joint
        state = pb.getLinkState(env.robot_id, env.joint_index['r_ankle_y'])
        rpos, rorn = state[4:6] # joint

        M = pb.getMatrixFromQuaternion(lorn) # orientation of ankle frame in world coordinates
        M = np.array(M).reshape(3,3)
        # print(M)
        dx = np.array(lposA) - np.array(lpos)
        loffset = np.dot(dx, M)
        # offset = np.dot(M, dx)
        print(loffset)

        M = pb.getMatrixFromQuaternion(rorn) # orientation of ankle frame in world coordinates
        M = np.array(M).reshape(3,3)
        # print(M)
        dx = np.array(rposA) - np.array(rpos)
        roffset = np.dot(dx, M)
        # offset = np.dot(M, dx)
        print(roffset)

        moffset = (np.fabs(loffset) + np.fabs(roffset)) / 2
        loffset = np.sign(loffset) * moffset
        roffset = np.sign(roffset) * moffset
        print(loffset)
        print(roffset)

        input('.')
    





