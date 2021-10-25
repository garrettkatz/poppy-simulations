from math import sin, cos
import itertools as it
import math
import sys
sys.path.append('../../envs')
import os
import pybullet as p
import random
from ergo_jr import PoppyErgoJrEnv
from object2urdf import ObjectUrdfBuilder

env = PoppyErgoJrEnv(p.POSITION_CONTROL)
fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/../../../urdfs/files'
file_list =[]
for file in os.listdir(fpath):
    if file.endswith(".urdf"):
        file_list.append(os.path.join(fpath,file))
print (file_list)
#random.shuffle(file_list)
#physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-1)
start_pos = [[0.08,0.08,0.3],[-0.08,0.08,0.3],[0.08,-0.08,0.3],[-0.08,-0.08,0.3]]
counter =0
Obj_list =[0,0,0,0]
for file in file_list:
    Obj_list[counter]=p.loadURDF(file,start_pos[counter],p.getQuaternionFromEuler([0,0,0]))
    counter=counter+1
    if counter == 4:
        break
for i in range(1000):
    p.stepSimulation()



#QA start
#x = rsin(θ), y = rcos(θ)
#let r = 0.08
#initial theta = ?
radius = 0.08
theta = 0
pos = [radius*math.sin(theta) , radius*math.cos(theta) , 0.2]
orn = [0,0,0]
targ = [pos,orn]
action = env.inverse_kinematics([5, 7],targ)
t1=p.getBasePositionAndOrientation(Obj_list[0])
t2=p.getBasePositionAndOrientation(Obj_list[1])
t3=p.getBasePositionAndOrientation(Obj_list[2])
t4=p.getBasePositionAndOrientation(Obj_list[3])