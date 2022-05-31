import numpy as np
import os, sys
import pybullet as pb

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv

def sample_goal(env):
    pos, orn, vel, ang = env.get_base()
    pos = np.array(pos) + np.random.uniform((-.2, 0, 0), (.2, .2, 0))
    orn = pb.getQuaternionFromEuler((0, 0, np.random.rand()*np.pi + np.pi/2))
    vel = np.array(vel) + np.random.randn(3) * np.array([0.1, 0.1, 0])
    ang = np.zeros(3)
    return tuple(map(tuple, (pos, orn, vel, ang)))

def goal_distance(base1, base2):
    base1 = np.concatenate(tuple(map(np.array, base1)))
    base2 = np.concatenate(tuple(map(np.array, base2)))
    return np.sum((base1 - base2)**2)

if __name__ == "__main__":

    # launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False)
    env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))
    input('.')

    # get initial base
    init_base = env.get_base()

    # sample a goal
    goal_base = sample_goal(env)
    env.set_base(*goal_base)
    input('.')

    env.close()

    # check distance
    print(goal_distance(init_base, goal_base))

