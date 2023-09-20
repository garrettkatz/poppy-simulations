# visualize grasp points in the voxel grid
# ax = pt.gcf().add_subplot(projection='3d')
# ax.voxels(voxels.astype(bool), alpha=0.5)
# pt.plot(*cands[0].T, marker='o', color='red')
# pt.show()
import MultObjPick
import pybullet as pb
from BaselineLearner import BaselineLearner
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
import numpy as np

grasp_width = 1  # distance between grippers in voxel units
voxel_size = 0.03  # dimension of each voxel
table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

learner = BaselineLearner(grasp_width, voxel_size)

exp = MultObjPick.experiment()
exp.CreateScene()
env = exp.env

# better view of tabletop
pb.resetDebugVisualizerCamera(
    cameraDistance=1.4,
    cameraYaw=-1.2,
    cameraPitch=-39.0,
    cameraTargetPosition=(0., 0., 0.),
)

dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
n_parts = 6
rgb = [(.75, .25, .25)] * n_parts
obj = MultObjPick.Obj(dims, n_parts, rgb)
obj.GenerateObject(dims, n_parts, [0, 0, 0])

obj2 = MultObjPick.Obj(dims, n_parts, rgb)
obj2.GenerateObject(dims, n_parts, [0, 0, 0])

Child = obj2.crossover(obj,obj2,3)
obj_id = exp.Spawn_Object(Child)
