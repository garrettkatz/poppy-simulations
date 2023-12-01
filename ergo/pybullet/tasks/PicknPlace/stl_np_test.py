# Before running this script, need to run: pip install --user stl-to-voxel

# stl-to-voxel parameters, higher resolution is slower, especially visualization
resolution = 20
voxel_size = None
parallel = False

import os
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from stltovoxel.convert import convert_mesh

# download Rubber_Duck.stl from : https://www.thingiverse.com/download:2336973
input_file_path = os.path.join(os.environ["HOME"], "Downloads", "Rubber_Duck.stl")

# based on https://github.com/cpederkoff/stl-to-voxel/blob/master/stltovoxel/convert.py
print("converting...")
mesh_obj = mesh.Mesh.from_file(input_file_path)
org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
vol, scale, shift = convert_mesh(org_mesh, resolution, voxel_size, parallel)

# duck happens to be oriented on its side, rotate upright
vol = np.rot90(vol, axes=(0,2))

# visualize, based on https://matplotlib.org/stable/gallery/mplot3d/voxels.html
print("\nvisualizing...")
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(vol, facecolors='y', edgecolor='k')
plt.axis("equal")
plt.show()

