import os, sys
import pybullet as pb
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv

from goals import sample_goal, goal_distance

if __name__ == "__main__":

    show = False

    traj_len = 2

    num_episodes = 10
    num_steps = 1

    s_sigma = 0.01
    a_sigma = 0.01

    # launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False, show=show)
    env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))

    num_inp = len(env.joint_index) + 26
    num_hid = 64
    num_out = traj_len*len(env.joint_index)

    net = tr.nn.Sequential(
        tr.nn.Linear(num_inp, num_hid),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(num_hid, num_out),
    )
    opt = tr.optim.SGD(net.parameters(), lr=0.001)

    rewards = np.empty((num_episodes, num_steps))
    sl_losses = np.empty((num_episodes, num_steps))

    for episode in range(num_episodes):

        env.reset()
        env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))

        for step in range(num_steps):

            old_base = env.get_base()
            g = sample_goal(env)
            sg = np.concatenate((env.get_position(),) + old_base + g)
            inp = sg + np.random.randn(len(sg)) * s_sigma

            out = net(tr.tensor(inp).float())
            dst = tr.distributions.normal.Normal(out, a_sigma)
            a = dst.sample()
            log_prob = dst.log_prob(a).sum()

            traj = a.detach().numpy().reshape((traj_len, -1))
            for target in traj: env.goto_position(target)
            new_base = env.get_base()

            reward = np.exp(-goal_distance(new_base, g))
            rl_loss = -reward * log_prob
            rl_loss.backward()

            sn = np.concatenate((env.get_position(),) + old_base + new_base)
            sl_loss = tr.sum((net(tr.tensor(sn).float()) - a.detach())**2)
            sl_loss.backward()

            opt.zero_grad()
            opt.step()

            rewards[episode, step] = reward
            sl_losses[episode, step] = sl_loss.item()

        print(f"ep {episode}: rl {rewards[episode].mean()}, sl {sl_losses[episode].mean()}")

    env.close()

    np.savez("babble.npz", rewards=rewards, sl_losses=sl_losses)

    pt.subplot(1,2,1)
    pt.plot(rewards)
    pt.subplot(1,2,2)
    pt.plot(sl_losses)
    pt.show()
            

