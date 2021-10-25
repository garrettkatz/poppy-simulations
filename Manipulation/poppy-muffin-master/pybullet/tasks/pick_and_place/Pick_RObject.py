from math import sin, cos
import itertools as it
import sys
sys.path.append('../../envs')
import os
import math
import pybullet as p
import numpy as np
import random
from ergo_jr import PoppyErgoJrEnv
import time
from object2urdf import ObjectUrdfBuilder

env = PoppyErgoJrEnv(p.POSITION_CONTROL)
fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/../../../urdfs/files/Pybullet_assets/random_urdfs/001'
file_list =[]
for file in os.listdir('D:/G/poppy-muffin-master/urdfs/Pybullet_assets/random_urdfs/001'):
    if file.endswith(".urdf"):
        file_list.append(os.path.join(fpath,file))
print (file_list)
def get_tip_targets(p, q, d):
    m = q
    t1 = p[0]-d*m[1], p[1]-d*m[4], p[2]-d*m[7]
    t2 = p[0]+d*m[1], p[1]+d*m[4], p[2]+d*m[7]
    return (t2, t1)

p.setGravity(0,0,-9)
start_pos = [[0.11,0.11,0.08],[-0.11,0.11,0.01],[0.11,-0.11,0.01],[-0.11,-0.11,0.01]]
counter =0
cubeStartPos = [0,-0.25,0]
cubeStartPos2 = [0.18,-0.20,0]
Obj_list =[0,0,0,0]
start_or = p.getQuaternionFromEuler([0,0,math.radians(270)])
start_or2 = p.getQuaternionFromEuler([0,0,math.radians(270+30)])
robottId2 = p.loadURDF("D:/G/poppy-muffin-master/urdfs/Pybullet_assets/random_urdfs/001/001.urdf", cubeStartPos , start_or)
robottId3 = p.loadURDF("D:/G/poppy-muffin-master/urdfs/Pybullet_assets/random_urdfs/001/001.urdf", cubeStartPos2 , start_or2)

print(p.getJointInfo(robottId3,1))
_link_name_to_index = {p.getBodyInfo(robottId3)[0].decode('UTF-8'): -1, }

for _id in range(p.getNumJoints(robottId3)):
    _name = p.getJointInfo(robottId3, _id)[12].decode('UTF-8')
    _link_name_to_index[_name] = _id
#move to first position
for i in range(50):

    pos = p.getLinkState(robottId2,1)
    quat = p.getMatrixFromQuaternion(pos[1])
    print(pos[0],pos[1])
    new_p = pos[0]
    new_o = pos[1]
    tar_args = get_tip_targets(new_p,quat,0.012)
    i_k = env.inverse_kinematics([5, 7],tar_args)
    env.goto_position(i_k,1)

    p.stepSimulation()

    time.sleep(1. / 240.)
#close gripper
for i in range(50):
    m = env.close_gripper(-0.22)
    p.stepSimulation()
    #time.sleep(1. / 100.)

#Move to new position
pos = p.getLinkState(robottId2, 1)
quat = p.getMatrixFromQuaternion(pos[1])
print(pos[0], pos[1])
new_p = pos[0]
newP_list = list(new_p)
newP_list[2]=newP_list[2]+0.1
new_p_tuple = tuple(newP_list)
new_o = pos[1]
tar_args = get_tip_targets(new_p_tuple, quat, 0.010)
i_k = env.inverse_kinematics([5, 7], tar_args)


env.goto_position(i_k, 1)
    #env.close_gripper(-0.22)

    #p.stepSimulation()
pos = p.getLinkState(robottId3, 1)
quat = p.getMatrixFromQuaternion(pos[1])
print(pos[0], pos[1])
new_p = pos[0]
newP_list = list(new_p)
newP_list[2]=newP_list[2]+0.15
new_p_tuple = tuple(newP_list)
new_o = pos[1]
tar_args = get_tip_targets(new_p_tuple, quat, 0.010)
i_k = env.inverse_kinematics([5, 7], tar_args)
env.goto_position(i_k, 1)


Errorlist=[]
'''for i in range(100):
    pos_bot_tuple = p.getLinkState(robottId2,1)
    pos_obj_tuple = p.getLinkState(robottId3,1)
    pos_bot_numpy = np.array(pos_bot_tuple[0])
    pos_obj_numpy = np.array(pos_obj_tuple[0])
    pos_obj_numpy[2]= pos_obj_numpy[2]+0.05
    Error_1 = np.sum((pos_bot_numpy - pos_obj_numpy) ** 2, axis=0)
    Errorlist.append(Error_1)
    direction =  pos_obj_numpy - pos_bot_numpy
    new_pos_bot = pos_bot_numpy+ 0.1*(direction)
    quat = p.getMatrixFromQuaternion(pos_obj_tuple[1])
    tar_args = get_tip_targets(new_pos_bot, quat, 0.010)
    i_k = env.inverse_kinematics_Orn([5, 7], tar_args,[pos_obj_tuple[1],pos_obj_tuple[1]])
    env.goto_position(i_k,1)
    #Error_2 = np.sum((p1 - p2) ** 2, axis=0)
'''
for i in range(100):
    pos_bot_tuple = p.getLinkState(robottId2,1)
    pos_obj_tuple = p.getLinkState(robottId3,1)
    pos_bot_numpy = np.array(pos_bot_tuple[0])
    pos_obj_numpy = np.array(pos_obj_tuple[0])
    pos_obj_numpy[2]= pos_obj_numpy[2]+0.05
    Error_1 = np.sum((pos_bot_numpy - pos_obj_numpy) ** 2, axis=0)
    Errorlist.append(Error_1)
    direction =  pos_obj_numpy - pos_bot_numpy
    new_pos_bot = tuple(pos_bot_numpy+ (direction))
    quat = p.getMatrixFromQuaternion(pos_bot_tuple[1])
    tar_args = get_tip_targets(new_pos_bot, quat, 0.010)
    i_k = env.inverse_kinematics([5, 7], tar_args)
    env.goto_position(i_k,1)


    pos_bot_tuple2 = p.getLinkState(robottId2, 0)
    pos_obj_tuple2 = p.getLinkState(robottId3, 0)
    pos_bot_numpy2 = np.array(pos_bot_tuple2[0])
    pos_obj_numpy2 = np.array(pos_obj_tuple2[0])
    pos_obj_numpy2[2] = pos_obj_numpy2[2] + 0.05
    Error_2 = np.sum((pos_bot_numpy2 - pos_obj_numpy2) ** 2, axis=0)
    #Errorlist.append(Error_2)
    direction2 = pos_obj_numpy2 - pos_bot_numpy2
    new_pos_bot2 = tuple(pos_bot_numpy2 + (direction))
    quat2 = p.getMatrixFromQuaternion(pos_bot_tuple2[1])
    tar_args2 = get_tip_targets(new_pos_bot2, quat2, 0.010)
    i_k2 = env.inverse_kinematics([5, 7], tar_args2)
    env.goto_position(i_k2,1)
    #Error_2 = np.sum((p1 - p2) ** 2, axis=0)


for i in range(20):
    m = env.close_gripper(0.22)
    p.stepSimulation()
for i in range(50):
    #m = env.close_gripper(0.22)
    p.stepSimulation()
    time.sleep(1. / 240.)

t1=p.getBasePositionAndOrientation(Obj_list[0])
#t2=p.getBasePositionAndOrientation(Obj_list[1])
#t3=p.getBasePositionAndOrientation(Obj_list[2])
#t4=p.getBasePositionAndOrientation(Obj_list[3])
print(Errorlist)
print("hey")
def Move_to(pos):
    #inverse kinematics and action
    return
def Main_Pickup(Obj_pos , Final_pos):
    return
    #choose object
    #moveto(object_pos -1)''
    #open gripper
    #moveto(object_pos)
    #close


    #---------Move Pickedup Object to another position ------
    #moveto(targetpos)
    #open gripper
    #moveto(targetpos[z]+1)