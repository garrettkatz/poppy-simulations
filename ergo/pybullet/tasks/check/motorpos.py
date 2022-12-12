"""
Do these commands first:
  git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
  cd pytorch_kinematics/
  pip install --user -e .
"""

import os, sys
import numpy as np
import pybullet as pb
import pytorch_kinematics as pk
import torch as tr

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

# copied from pytorch kinematics tests
def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot

if __name__ == "__main__":

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=True)

    # get differentiable kinematic chain
    chain = pk.build_chain_from_urdf(open(os.path.join("..", "..", "..", "urdfs", "ergo", "poppy_ergo.pybullet.urdf")).read())
    chain.to(dtype=tr.float64)
    th = [0] * len(chain.get_joint_parameter_names(exclude_fixed=True))
    ret = chain.forward_kinematics(th)
    for name, tg in ret.items():
        pos, rot = quat_pos_from_transform3d(tg)
        print(f"{name}: {pos.numpy()}, {rot.numpy()}")


    print(chain.get_joint_parameter_names(exclude_fixed=False), len(chain.get_joint_parameter_names(exclude_fixed=False)))
    print(env.joint_name.values(), len(env.joint_name))

    input('.')

    env.set_position(np.zeros(env.num_joints))
    env.step()

    pos = np.zeros((env.num_joints, 3))
    for idx in range(env.num_joints):
        state = pb.getLinkState(env.robot_id, idx)
        pos[idx] = state[0]
        print(f"{idx}: {env.joint_name[idx]}: {tuple(pos[idx])}")

        pb.changeVisualShape(
            env.robot_id,
            idx,
            rgbaColor = (1, 1, 1, .5),
        )
        pb.addUserDebugPoints(
            pointPositions = (tuple(pos[idx]),),
            pointColorsRGB = ((1,0,0),),
            pointSize=40,
        )
        pb.addUserDebugText(
            env.joint_name[idx],
            textPosition = tuple(pos[idx]),
            textColorRGB = (0,0,0),
            textSize=1,
        )
    print(pos.mean(axis=0))

    input('.')
    


