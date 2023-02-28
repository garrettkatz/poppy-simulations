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

    # # # toe
    # # angles[env.joint_index['r_hip_y']] = -np.pi/2
    # # angles[env.joint_index['l_hip_y']] = -np.pi/2
    # # env.set_base(orn=pb.getQuaternionFromEuler((3*np.pi/4, 0, 0)))
    # env.set_base(pos=(0,0,.5), orn=pb.getQuaternionFromEuler((1, 0, 0)))

    # # heel
    # angles[env.joint_index['r_hip_y']] = -np.pi/2
    # angles[env.joint_index['l_hip_y']] = -np.pi/2
    # env.set_base(orn=pb.getQuaternionFromEuler((1*np.pi/4, 0, 0)))
    env.set_base(pos=(0,0,.5), orn=pb.getQuaternionFromEuler((-1, 0, 0)))

    # # sole
    # pass

    env.set_position(angles)

    while True:
        env.step(action=angles)    

        # left_pts = pb.getContactPoints(env.robot_id, linkIndexA = env.joint_index['l_ankle_y'])
        # right_pts = pb.getContactPoints(env.robot_id, linkIndexA = env.joint_index['r_ankle_y'])
        # # for p,pt in enumerate(pts):
        # #     posA, posB, normB, dist, force, = pt[5:10]
        # #     print(f"pt {p}:")
        # #     print(f"posA = {posA}")
        # #     print(f"posB = {posB}")
        # #     print(f"normB = {normB}")
        # #     print(f"dist = {dist}")
        # #     print(f"force = {force}")

        left_pts = pb.getClosestPoints(env.robot_id, env.ground_id, distance=10, linkIndexA=env.joint_index['l_ankle_y'])
        right_pts = pb.getClosestPoints(env.robot_id, env.ground_id, distance=10, linkIndexA=env.joint_index['r_ankle_y'])

        print(len(left_pts), len(right_pts))
        # input('.')

        pts = left_pts
        if min(len(left_pts), len(right_pts)) > 0:
            lposA, posB, normB, dist, force, = left_pts[0][5:10]
            rposA, posB, normB, dist, force, = right_pts[0][5:10]
        else:
            continue

        jnt_loc = env.forward_kinematics()
        print('l/r ankle jnt loc')
        print(jnt_loc[env.joint_index['l_ankle_y']])
        print(jnt_loc[env.joint_index['r_ankle_y']])

        pb.resetDebugVisualizerCamera(
            cameraDistance = 1,
            cameraYaw = 0,
            cameraPitch = 0,
            cameraTargetPosition = (0, 0, 0),
        )

        state = pb.getLinkState(env.robot_id, env.joint_index['l_ankle_y'])
        lpos, lorn = state[4:6] # joint
        state = pb.getLinkState(env.robot_id, env.joint_index['r_ankle_y'])
        rpos, rorn = state[4:6] # joint

        print('l/r offsets')

        M = pb.getMatrixFromQuaternion(lorn) # orientation of ankle frame in world coordinates
        M = np.array(M).reshape(3,3)
        # print(M)
        dx = np.array(lposA) - np.array(lpos)
        loffset = np.dot(dx, M)
        # offset = np.dot(M, dx)
        print(loffset)

        for j in range(3):
            pb.addUserDebugLine(lineFromXYZ=lpos, lineToXYZ=np.array(lpos) + 0.05*M[:,j], lineColorRGB=np.eye(3)[j])

        pb.addUserDebugLine(lineFromXYZ=lposA - 0.01*M[:,0], lineToXYZ=np.array(lposA) + 0.01*M[:,0], lineColorRGB=np.zeros(3))
        pb.addUserDebugLine(lineFromXYZ=lposA - 0.01*M[:,1], lineToXYZ=np.array(lposA) + 0.01*M[:,1], lineColorRGB=np.zeros(3))

        M = pb.getMatrixFromQuaternion(rorn) # orientation of ankle frame in world coordinates
        M = np.array(M).reshape(3,3)
        # print(M)
        dx = np.array(rposA) - np.array(rpos)
        roffset = np.dot(dx, M)
        # offset = np.dot(M, dx)
        print(roffset)

        for j in range(3):
            pb.addUserDebugLine(lineFromXYZ=rpos, lineToXYZ=np.array(rpos) + 0.05*M[:,j], lineColorRGB=np.eye(3)[j])

        pb.addUserDebugLine(lineFromXYZ=rposA - 0.01*M[:,0], lineToXYZ=np.array(rposA) + 0.01*M[:,0], lineColorRGB=np.zeros(3))
        pb.addUserDebugLine(lineFromXYZ=rposA - 0.01*M[:,1], lineToXYZ=np.array(rposA) + 0.01*M[:,1], lineColorRGB=np.zeros(3))

        print('averaged')

        moffset = (np.fabs(loffset) + np.fabs(roffset)) / 2
        loffset = np.sign(loffset) * moffset
        roffset = np.sign(roffset) * moffset
        print(loffset)
        print(roffset)

        input('.')
    





