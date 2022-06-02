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

    show_train = False
    show_test = True

    # dotrain = False
    dotrain = True

    traj_len = 2
    duration = 0.25

    num_updates = 1000
    num_episodes = 20
    num_steps = 1
    save_period = 5

    s_sigma = 0.01
    a_sigma = 0.01
    learning_rate = 0.0001

    num_joints = 36
    num_inp = num_joints + 26
    num_hid = 128
    num_out = traj_len*num_joints

    net = tr.nn.Sequential(
        tr.nn.Linear(num_inp, num_hid),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(num_hid, num_out),
    )

    if dotrain:

        opt = tr.optim.SGD(net.parameters(), lr=learning_rate)
        # opt = tr.optim.Adam(net.parameters(), lr=learning_rate)
    
        rewards = np.empty((num_updates, num_episodes, num_steps))
        sl_losses = np.empty((num_updates, num_episodes, num_steps))

        for update in range(num_updates):

            for episode in range(num_episodes):

                # launches the simulator
                env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False, show=show_train)
                env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))

                for step in range(num_steps):
        
                    old_base = env.get_base()
                    old_joints = env.get_position()
                    g = sample_goal(env)

                    sg = np.concatenate((old_joints,) + old_base + g)
                    inp = sg + np.random.randn(len(sg)) * s_sigma        
                    out = net(tr.tensor(inp).float())
                    dst = tr.distributions.normal.Normal(out, a_sigma)
                    a = dst.sample()
                    log_prob = dst.log_prob(a).sum()
        
                    traj = a.detach().numpy().reshape((traj_len, -1))
                    for target in traj: env.goto_position(target, duration=duration)
                    new_base = env.get_base()
        
                    reward = np.exp(-goal_distance(new_base, g))
                    # reward =  1 / (1 + goal_distance(new_base, g))
                    rl_loss = -reward * log_prob
                    rl_loss.backward()
        
                    sn = np.concatenate((old_joints,) + old_base + new_base)
                    sl_loss = tr.sum((net(tr.tensor(sn).float()) - a.detach())**2)
                    sl_loss.backward()
        
                    rewards[update, episode, step] = reward
                    sl_losses[update, episode, step] = sl_loss.item()

                env.close()

            opt.step()
            opt.zero_grad()
    
            if update % save_period == 0:
                np.savez("babble.npz", rewards=rewards, sl_losses=sl_losses)
                tr.save(net.state_dict(), "babble.pt")

            print(f"{update}/{num_updates}: rl {rewards[update].mean():e}, sl {sl_losses[update].mean():e}")
            
        np.savez("babble.npz", rewards=rewards, sl_losses=sl_losses)
        tr.save(net.state_dict(), "babble.pt")

    if show_test:
    
        npz = np.load("babble.npz")
        rewards, sl_losses = npz["rewards"], npz["sl_losses"]
    
        pt.subplot(1,2,1)
        pt.plot(rewards.reshape((num_updates, -1)).mean(axis=1))
        pt.subplot(1,2,2)
        pt.plot(sl_losses.reshape((num_updates, -1)).mean(axis=1))
        pt.show()
    
        net.load_state_dict(tr.load("babble.pt"))
        net.eval()
    
        env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=False, show=True)
        # env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=True)
        env.set_base(orn = pb.getQuaternionFromEuler((0,0,np.pi)))
    
        for step in range(num_steps):
    
            old_base = env.get_base()
            g = sample_goal(env)
            sg = np.concatenate((env.get_position(),) + old_base + g)
            a = net(tr.tensor(sg).float())
    
            traj = a.detach().numpy().reshape((traj_len, -1))
            for target in traj: env.goto_position(target, duration=duration)
    
        env.close()
    
    
