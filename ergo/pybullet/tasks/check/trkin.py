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
import matplotlib.pyplot as pt

# copied from pytorch kinematics tests
def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot

if __name__ == "__main__":

    # get differentiable kinematic chain
    chain = pk.build_chain_from_urdf(open(os.path.join("..", "..", "..", "urdfs", "ergo", "poppy_ergo.pybullet.urdf")).read())
    chain.to(dtype=tr.float64)
    joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
    angles = tr.zeros(1, len(joint_names), requires_grad=True)

    # print(chain._root)

    eta = 0.1
    num_iters = 1000

    fig = pt.figure()
    ax = fig.add_subplot(projection='3d')
    pt.ion()
    pt.show()

    for itr in range(num_iters):
        FK = chain.forward_kinematics(angles)
        pos = []
        for name, tg in FK.items():
            pos_tg, rot_tg = quat_pos_from_transform3d(tg)
            # print(f"{name}: {pos.numpy()}, {rot.numpy()}")
            # print(f"{name}: {pos_tg}")
            pos.append(pos_tg.squeeze(0))
        pos = tr.stack(pos)
        cop = pos.mean(dim=0)
    
        xy_err = tr.sum(cop[:2]**2)
        z_err = -cop[2]
        err = xy_err + z_err
        
        err.backward()
        angles.data -= eta * angles.grad
        angles.grad *= 0

        if itr % 10 == 0:
            print(f"{itr}: {xy_err.item()} + {z_err.item()} = {err.item()}")

            pt.cla()
            npos = pos.detach().numpy()
            ax.scatter(*npos.T)
            for name, tg in FK.items():
                src_pos = FK[name].get_matrix()[:, :3, 3].detach().numpy()
                frame = chain.find_frame(name + "_frame")
                for child in frame.children:
                    child_name = child.name[:-len("_frame")]
                    dst_pos = FK[child_name].get_matrix()[:, :3, 3].detach().numpy()
                    line_data = np.concatenate((src_pos, dst_pos), axis=0)
                    ax.plot(*line_data.T, color='b')                    
            pt.axis('equal')
            pt.pause(0.01)

