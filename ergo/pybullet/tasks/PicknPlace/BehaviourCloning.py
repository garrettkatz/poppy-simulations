import torch
import torch.nn as nn
import numpy as np
import sys,os
import pickle
import glob, os
import re

import matplotlib.pyplot as plt
sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

print(try_gpu())

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(78,256)
        self.lin2 = nn.Linear(256,1024)
        self.fc = nn.Linear(1024,168) #reshape 168 into 4x42
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        l1 = self.lin1(x)
        l2 = self.relu(self.lin2(l1))
        out = self.fc(l2)
        return out

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(36, 128)  # Relative pos of all voxels of object
        self.lin2 = nn.Linear(36, 128)    # Absolute position of all voxels of object
        self.lin3 = nn.Linear(6,100)   # Grip points flatten , xyz for 2 fingers
        self.fc = nn.Linear(356, 168)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, ip_angles,ip_graspVoxel,ip_Object):
        l1 = self.lin1(ip_angles.float())
        l2 = self.lin2(ip_graspVoxel.float())
        l3 = self.lin3(ip_Object.float())
        l4 = np.concat(l1,l2,l3)
        out = self.fc(l4)
        return out

def flatten(xss):
    return [x for xs in xss for x in xs]
import itertools
import matplotlib.pyplot as plt

Network =Model_1()
import torch.optim as optim
from operator import add
Obj_pos_abs = []
criterion= nn.MSELoss()
optimizer =optim.Adam(Network.parameters(),lr=0.001)

if __name__ == "__main__":
    f = open(f'Bh_dataset_1_graspPoint.pickle', 'rb')
    data1 = pickle.load(f)
    f.close()

    f = open(f'Bh_dataset_1_OBJ_voxels.pickle', 'rb')
    data2 = pickle.load(f)
    f.close()

    f = open(f'Bh_dataset_1_objdetails.pickle', 'rb')
    data3 = pickle.load(f)
    f.close()

    f = open(f'Bh_dataset_1_Trajectory.pickle', 'rb')
    data4 = pickle.load(f)
    f.close()
    LossList = []
    for i in range(len(data1)):
        for k in range(len(data1[i])):
            Grip_pos = flatten(data1[i][k])
            Rel_voxel_pos = data2[i][k]
            Obj_pos_list = list(data3[i][k])
            Abs_voxel_pos = data2[i][k]
            for j in range(len(Rel_voxel_pos)):
                Abs_voxel_pos[j][0] = Abs_voxel_pos[j][0] +  Obj_pos_list[0]
                Abs_voxel_pos[j][1] = Abs_voxel_pos[j][1] + Obj_pos_list[1]
                Abs_voxel_pos[j][2] = Abs_voxel_pos[j][2] + Obj_pos_list[2]

            r_vox_p = flatten(Rel_voxel_pos)
            a_vox_p = flatten(Abs_voxel_pos)
            Traj = (data4[i][k])
          #  Input = [n for n in (Grip_pos, a_vox_p, r_vox_p)]
            Input = list(itertools.chain(Grip_pos, a_vox_p, r_vox_p))
            print("data ready for this epoch")
            Input_t = torch.Tensor(Input)
            Output = Network(Input_t)
            Target = torch.Tensor(flatten(Traj))
            loss = criterion(Output,Target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            LossList.append(loss)

            Network.zero_grad()
            loss.backward()

    print("done")

    x_axis = np.arange(2501)
    y_axis = LossList
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)
    plt.show()

    print("done")
        


