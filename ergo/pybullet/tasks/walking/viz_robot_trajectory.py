import os, sys, time
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv

import base_trajectory as bt

if __name__ == "__main__":

    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False, show=True)
    pos, _, _, _ = env.get_base()
    z = pos[2]

    # rotation about z by pi/2 faces Poppy along x axis
    a0 = np.pi/2

    duration = 5
    num_steps = int(duration / (env.timestep * env.control_period))
    print(f"{num_steps} steps")

    y0 = np.array([0, 0])
    dy0 = np.array([0, 1])
    y1, dy1 = bt.get_goal()
    Y, t = bt.get_traj(y0, y1, dy0, dy1, num_steps)
    dY = Y[:,1:] - Y[:,:-1]

    env.set_base(orn = pb.getQuaternionFromEuler((0,0,a0 + np.arctan2(dy0[1], dy0[0]))))

    input('.')


    for step in range(num_steps-1):

        env.set_base(
            pos = (Y[0,step], Y[1,step], z),
            orn = pb.getQuaternionFromEuler((0,0,a0 + np.arctan2(dY[1,step], dY[0,step])))
        )

        pb.stepSimulation()
        time.sleep(env.timestep)

    env.close()
