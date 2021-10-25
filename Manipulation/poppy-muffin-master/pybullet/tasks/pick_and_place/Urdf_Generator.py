from object2urdf import ObjectUrdfBuilder

import os
import sys
import pybullet as p
import numpy as np

obj_folder = "../../urdfs/objects"

#builder.build_urdf(filename="D:/G/poppy-muffin-master/urdfs/test/ducksie.stl", force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'mass')
#builder = ObjectUrdfBuilder("D:/G/poppy-muffin-master/urdfs/test")
#builder.build_library(force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'top')
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
obj_urdf ="D:/G/poppy-muffin-master/urdfs/test/ducksie.stl.urdf"
boxStartPos = [0.5, 0.5, 1.5]

boxId = p.loadURDF(obj_urdf, boxStartPos)

while True:
    p.stepSimulation()