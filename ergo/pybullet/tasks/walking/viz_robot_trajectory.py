import os, sys, time
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv

import base_trajectory as bt

if __name__ == "__main__":

    # rotation about z by pi/2 faces Poppy along x axis
    a0 = np.pi/2

    y0 = np.array([0, 0])
    dy0 = np.array([0, 1])
    y1, dy1 = bt.get_goal()
    Y, t = bt.get_traj(y0, y1, dy0, dy1, 100)
    dY = Y[:,1:] - Y[:,:-1]

    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False, show=True)
    pos, _, _, _ = env.get_base()
    z = pos[2]
    env.set_base(orn = pb.getQuaternionFromEuler((0,0,a0 + np.arctan2(dy0[1], dy0[0]))))

    input('.')


    for step in range(99):

        env.set_base(
            pos = (Y[0,step], Y[1,step], z),
            orn = pb.getQuaternionFromEuler((0,0,a0 + np.arctan2(dY[1,step], dY[0,step])))
        )

        pb.stepSimulation()
        time.sleep(0.1)

    env.close()
