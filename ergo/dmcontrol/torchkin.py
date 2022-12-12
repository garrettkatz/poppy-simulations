"""
Do these commands first:
  git clone https://github.com/UM-ARM-Lab/pytorch_kinematics.git
  cd pytorch_kinematics/
  pip install --user -e .
"""

import os, sys
import numpy as np
import pytorch_kinematics as pk
import torch as tr
from dm_control import mjcf

# copied from pytorch kinematics tests
def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot

if __name__ == "__main__":

    # get differentiable kinematic chain

    mjcf_path = os.path.join("..","urdfs","ergo","poppy_ergo.dmcontrol.mod.xml")
    chain = pk.build_chain_from_mjcf_file(mjcf_path)
    chain.to(dtype=tr.float64)
    th = [0] * len(chain.get_joint_parameter_names(exclude_fixed=True))
    ret = chain.forward_kinematics(th)
    for name, tg in ret.items():
        pos, rot = quat_pos_from_transform3d(tg)
        print(f"{name}: {pos.numpy()}, {rot.numpy()}")


