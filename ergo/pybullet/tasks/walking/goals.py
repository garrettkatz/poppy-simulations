import numpy as np
import os, sys
import pybullet as pb

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv

def sample_goal(env):

    # uniform new position in small circle in front of robot
    R = 0.1
    r = np.random.rand() * R
    th = np.random.rand() * 2*np.pi
    dx, dy = r*np.cos(th), r*np.sin(th) - R
    dpos = (dx, dy, 0)

    # orientation aligned with change in position
    dorn = pb.getQuaternionFromEuler((0, 0, np.arctan2(dy, dx)))

    # transform relative to current robot position
    bpos, born, _, _ = env.get_base()
    pos, orn = pb.multiplyTransforms(bpos, born, dpos, dorn)

    # velocity aligned with change in position
    vel = np.array(pos) - np.array(bpos)

    # normalize max speed proportional to change in position and at most 0.1
    vel *= np.random.rand() * 0.1 / (2*R)
    
    # no angular velocity
    ang = (0,)*3

    return pos, orn, tuple(vel), ang

def goal_distance(base1, base2):
    # base1 = np.concatenate(tuple(map(np.array, base1)))
    # base2 = np.concatenate(tuple(map(np.array, base2)))
    # return np.sum((base1 - base2)**2)

    pos1, orn1, vel1, ang1 = base1
    pos2, orn2, vel2, ang2 = base1
    dorn = pb.getDifferenceQuaternion(orn1, orn2)
    _, angle = pb.getAxisAngleFromQuaternion(dorn)
    base1 = np.concatenate((pos1, vel1, ang1))
    base2 = np.concatenate((pos2, vel2, ang2))
    return np.sum((base1 - base2)**2) + angle**2

if __name__ == "__main__":

    # launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False)
    env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))
    base = env.get_base()

    # visualize goal distribution
    import matplotlib.pyplot as pt
    
    pt.plot([.1,.1,-.1,-.1,.1], [.1,-.1,-.1,.1,.1], 'k--')
    dists = []
    for samp in range(200):
        goal = sample_goal(env)
        pos, orn, vel, ang = goal
        dists.append(goal_distance(base, goal))
        pt.plot([pos[0]], [pos[1]], 'ko')
        pt.arrow(pos[0], pos[1], vel[0], vel[1], color='b')

    print(f"average goal dist from initial base: {np.mean(dists)}")

    pt.show()


    # get initial base
    init_base = env.get_base()

    # sample a goal
    goal_base = sample_goal(env)
    env.set_base(*goal_base)
    input('.')

    env.close()

    # check distance
    print(goal_distance(init_base, goal_base))

