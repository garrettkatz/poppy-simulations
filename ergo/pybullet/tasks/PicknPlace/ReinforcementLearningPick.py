import os, sys
import numpy as np
import pybullet as pb

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))
sys.path.append(os.path.join('..', '..', 'objects'))
sys.path.append(os.path.join('..', 'walking'))
sys.path.append(os.path.join('..', 'seeing'))
sys.path.append(os.path.join('..', 'grasping'))
from torch import distributions as dist

from ergo import PoppyErgoEnv
import tabletop as tt

# integration APIs
import motor_control as mc
import vision as vz
import grasping as gr

import math as m


def get_tip_targets(p, q, d):
    m = q
    t1 = p[0] - d * m[0], p[1] - d * m[3], p[2] - d * m[6]
    t2 = p[0] + d * m[0], p[1] + d * m[3], p[2] + d * m[6]
    return (t1, t2)


def get_tip_targets2(p, q, d):
    m = q
    t1 = p[0] - d * m[1], p[1] - d * m[4], p[2] - d * m[7]
    t2 = p[0] + d * m[1], p[1] + d * m[4], p[2] + d * m[7]
    return (t1, t2)

def slot_arrays(inn, out):
    # inn/out: inner/outer half extents, as np arrays
    mid = (inn + out) / 2
    ext = (out - inn) / 2
    positions = np.array([
        [-mid[0], 0, 0],
        [+mid[0], 0, 0],
        [0, -mid[1], 0],
        [0, +mid[1], 0],
    ])
    extents = [
        (ext[0], out[1], out[2]),
        (ext[0], out[1], out[2]),
        (inn[0], ext[1], out[2]),
        (inn[0], ext[1], out[2]),
    ]

    return positions, extents


def make_state(angles,objectpos,ObjOrientation):
    objp = tr.tensor(objectpos[0])
    objo = tr.tensor(ObjOrientation)
    rangles = tr.tensor(angles)
    state_rep = tr.cat((rangles,objp,objo))
    return state_rep



from operator import add

def rewards(env , objpos):
    robotpos = env.get_position()
    lft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
    lmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
    rft_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
    rmt_pos = list(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
    rh_pos = tr.mul(tr.tensor(list(map(add,rft_pos,rmt_pos))),0.5)
    lh_pos = tr.mul(tr.tensor(list(map(add,lft_pos,lmt_pos))),0.5)
    rh_euc_distance = sum((rh_pos - tr.tensor(objpos[0]))**2)

    lh_euc_distance = sum((lh_pos - tr.tensor(objpos[0]))**2)
    closestarm = rh_euc_distance   #min(rh_euc_distance,lh_euc_distance)
    rew = 1 - 10*closestarm
    # more rewards needed
    return rew

def actionsForState(state):
    actionList = list()
    # 16 angles
    #populate with all possible actions
    return actionList

def NextState(State,action):
    new_state = State
    return new_state

import torch as tr
import math
class Policy_network(tr.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = tr.nn.Linear(48, 36)
        self.fc2 = tr.nn.Linear(36, 9)

        self.relu = tr.nn.LeakyReLU()
        self.tanh = tr.nn.Tanh()
        self.sig = tr.nn.Sigmoid() #tanh

    def forward(self, newstate):
        residual = newstate[27:36]
        outputp = self.fc1(newstate)
        outputp = self.fc2(self.relu(outputp))

        outputp = self.tanh(outputp)
        outputp = tr.mul(outputp,tr.pi/6)
        output = outputp.add(residual)
        return output.flatten()

    # remove residual
    # tune learning rate and other hyperparameters
    #

import random
Network =Policy_network()
import torch.optim as optim
criterion= tr.nn.BCELoss()
optimizer =optim.Adam(Network.parameters(),lr=0.01)

if __name__ == "__main__":

    # this launches the simulator
    for loop in range(5):
        LoopList = list()
        loss_list = list()
        X_axis_plot = list()
        for epoch in range(10000):
            X_axis_plot.append(epoch)
            # this adds the table
            env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)

            tt.add_table()
            angles2 = env.angle_dict(env.get_position())
            pos = pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])
            #info = pb.getLinkStates(1,env.joint_index["l_fixed_tip"])

            # cube_pos = pb.getBasePositionAndOrientation(3)
            angles2.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})

            Error_list = []
            Error_dist_list = []
            X_axis_plot = []
            Y_axis_plot = []
            # slot

            dims = np.array([.02, .04, .1])
            pos, ext = slot_arrays(dims / 2, dims / 2 + np.array([.01, .01, 0]))
            rgb = [(.75, .25, .25)] * 4
            boxes = list(zip(map(tuple, pos), ext, rgb))
            env.set_position(env.angle_array(angles2))


            slot = tt.add_box_compound(boxes)
            slot2 = tt.add_box_compound(boxes)

            t_pos = tt.table_position()
            t_ext = tt.table_half_extents()
            s_pos = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2) + np.random.randn(3) * np.array(
                [0.05, .02, 0])
            s_pos_2 = list(s_pos)
            s_pos_2[0] = s_pos_2[0] - 0.1
            pb.resetBasePositionAndOrientation(slot, s_pos, (0.0, 0.0, 0.0, 1))
            pb.resetBasePositionAndOrientation(slot2, tuple(s_pos_2), (0.0, 0.0, 0.0, 1))

            d_pos = (s_pos[0], s_pos[1], s_pos[2] + dims[2] / 2)
            d_pos_2 = (s_pos_2[0], s_pos_2[1], s_pos_2[2] + dims[2] / 2)
            d_ext = (dims[0] / 4, dims[1] / 4, dims[2] * 0.75)
            disk = tt.add_Obj_compound(d_pos, d_ext, (0, 0, 1))
            X_axis_plot.append(d_pos[0])
            Y_axis_plot.append(d_pos[1])
            angles = env.get_position()
            exit_counter = 0
            #while numattempts<100:
            running_loss =0.0
            prob_action = list()
            rew = list()
            while True: #not fail
                exit_counter = exit_counter+1
                if exit_counter > 100:
                    break

                state_angles_check = env.get_position()

                #state_angles = env.angle_dict(env.get_position())
                state_position = pb.getBasePositionAndOrientation(disk)
                state_quat = pb.getMatrixFromQuaternion(pb.getBasePositionAndOrientation(disk)[1])
                newstate = make_state(state_angles_check,state_position,state_quat)
                optimizer.zero_grad()
                probs = Network(newstate.float())
                m = dist.multivariate_normal.MultivariateNormal(probs,tr.mul(tr.eye(9),0.001))
                #m= tr.distributions.multivariate_normal()
                    #m = Categorical(probs)
                action = m.sample()
                new_angles = env.get_position()
                new_angles[27:36] = action
                next_state = env.goto_position(list(new_angles),1)
                #loss = -m.log_prob(action) * rewards(env,pb.getBasePositionAndOrientation(disk))
                prob_action.append(-m.log_prob(action))
                rew.append(rewards(env,pb.getBasePositionAndOrientation(disk)))
            # rew = rew - avg(rew)
            loss = sum(prob_action) * sum(rew)
            loss.backward()
            loss_list.append(sum(rew))
            optimizer.step()
            print(f'epoch [{epoch + 1},Training loss]: {loss:.3f}')
            pb.removeBody(disk)
            pb.removeBody(slot2)
            pb.removeBody(slot)
            env.close()

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1)
        fig.suptitle('Error Plot')
        axes[0].set(xlabel='Epoch - X Coordinate', ylabel='Loss')
        axes[0].scatter(X_axis_plot, loss_list)

        plt.show()
        plt.savefig("ErrorPlot"+loop+".png")
        input('.')
