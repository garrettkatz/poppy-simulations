import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def Voxelize(filename):
    # Load OBJ
    mesh = trimesh.load(filename)
    # Voxelize with trimesh (fallback)
    voxels = mesh.voxelized(pitch=15.0).matrix  # Larger pitch = fewer voxels
    print("Voxel grid shape:", voxels.shape)  # Should be (X,Y,Z)
    # 2. Get occupied voxel coordinates (shape: Nx3)
    occupied_voxels = np.argwhere(voxels)  # [[x1,y1,z1], [x2,y2,z2], ...]

    # 3. Normalize to origin (subtract min coordinates)
    min_coords = np.min(occupied_voxels, axis=0)  # Smallest x, y, z
    normalized_voxels = occupied_voxels - min_coords  # Shift to (0,0,0)

    # 4. Convert to list of 3D tuples
    coordinates = [tuple(coord) for coord in normalized_voxels]
    #first_coord = coordinates[0]  # (0, 1, 0)
    #normalized = [    (x - first_coord[0], y - first_coord[1], z - first_coord[2])    for x, y, z in coordinates]
    min_x = min(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    min_z = min(coord[2] for coord in coordinates)
    normalized = [(x - min_x, y - min_y, z - min_z) for x, y, z in coordinates]
    print(normalized)
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, facecolors='blue', edgecolor='none')
    #plt.show()
    return normalized


v=Voxelize("F6.obj")
print(v)
