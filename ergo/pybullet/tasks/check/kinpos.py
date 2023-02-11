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
    angles[env.joint_index['r_shoulder_x']] = -np.pi/6
    angles[env.joint_index['l_shoulder_x']] = +np.pi/6

    angles[env.joint_index['r_ankle_y']] = -np.pi/6
    angles[env.joint_index['l_ankle_y']] = +np.pi/6

    env.set_position(angles)
    env.step()

    com_pos = np.zeros((env.num_joints, 3))
    jnt_pos = np.zeros((env.num_joints, 3))
    for idx in range(env.num_joints):
        state = pb.getLinkState(env.robot_id, idx)
        com_pos[idx] = state[0]
        jnt_pos[idx] = state[4]
        print(f"{idx} {env.joint_name[idx]}: com={tuple(com_pos[idx])}, jnt={tuple(jnt_pos[idx])}")

        pb.addUserDebugText(
            env.joint_name[idx],
            textPosition = tuple(jnt_pos[idx]),
            textColorRGB = (0,0,0),
            textSize=1,
        )
        pb.addUserDebugLine(
            lineFromXYZ=jnt_pos[idx],
            lineToXYZ=com_pos[idx],
            lineColorRGB=(0,1,0),
        )

    for idx in range(-1, env.num_joints):
        pb.changeVisualShape(
            env.robot_id,
            idx,
            rgbaColor = (1, 1, 1, .5),
        )

    pb.addUserDebugPoints(
        pointPositions = jnt_pos,
        pointColorsRGB = np.tile(np.array([1,0,0]), (env.num_joints,1)),
        pointSize=40,
    )
    pb.addUserDebugPoints(
        pointPositions = com_pos,
        pointColorsRGB = np.tile(np.array([0,0,1]), (env.num_joints,1)),
        pointSize=40,
    )

    input('.')
    




