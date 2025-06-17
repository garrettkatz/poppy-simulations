import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from RL_discrete_mutations import is_connected
from matplotlib.colors import to_hex

def compute_density_field(obj, resolution=0.01, range_limit=0.03):
    positions = np.array(obj.positions)
    connected_scores = []
    coord_grid = []

    deltas = np.arange(-range_limit, range_limit + resolution, resolution)
    for idx, pos in enumerate(positions):
        for dx in deltas:
            for dy in deltas:
                for dz in deltas:
                    if dx == dy == dz == 0:
                        continue
                    new_positions = positions.copy()
                    new_positions[idx] = pos + np.array([dx, dy, dz])
                    if is_connected(new_positions):
                        connected_scores.append(1)
                    else:
                        connected_scores.append(0)
                    coord_grid.append((pos[0] + dx, pos[1] + dy, pos[2] + dz))

    return np.array(coord_grid), np.array(connected_scores)

def draw_voxel(ax, center, voxel_size=0.015, color='lightgray'):
    """Draws a cube centered at `center` with edge length `voxel_size`."""
    r = voxel_size / 2
    x, y, z = center
    xx = [x - r, x + r]
    yy = [y - r, y + r]
    zz = [z - r, z + r]
    for s, e in [
        ([xx[0], yy[0], zz[0]], [xx[1], yy[0], zz[0]]),
        ([xx[0], yy[0], zz[0]], [xx[0], yy[1], zz[0]]),
        ([xx[0], yy[0], zz[0]], [xx[0], yy[0], zz[1]]),
        ([xx[1], yy[1], zz[1]], [xx[0], yy[1], zz[1]]),
        ([xx[1], yy[1], zz[1]], [xx[1], yy[0], zz[1]]),
        ([xx[1], yy[1], zz[1]], [xx[1], yy[1], zz[0]]),
        ([xx[0], yy[1], zz[0]], [xx[0], yy[1], zz[1]]),
        ([xx[1], yy[0], zz[0]], [xx[1], yy[0], zz[1]]),
        ([xx[0], yy[0], zz[1]], [xx[1], yy[0], zz[1]]),
        ([xx[0], yy[0], zz[1]], [xx[0], yy[1], zz[1]]),
        ([xx[0], yy[1], zz[0]], [xx[1], yy[1], zz[0]]),
        ([xx[1], yy[0], zz[0]], [xx[1], yy[1], zz[0]]),
    ]:
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color, alpha=0.8)

def visualize_voxel_object(positions, colors=None, voxel_size=0.015):
    """Render the object using filled colored voxels."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    half = voxel_size / 2
    dx = dy = dz = voxel_size

    if colors is None:
        colors = ['gray'] * len(positions)

    for pos, color in zip(positions, colors):
        x, y, z = pos
        ax.bar3d(x - half, y - half, z - half, dx, dy, dz, color=color, shade=True, alpha=0.9)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Filled Voxel Object')
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

def visualize_density(coords, scores):
    coords = np.array(coords)
    scores = np.array(scores)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=scores, cmap='viridis')
    plt.colorbar(sc, ax=ax, label="Connectivity Score (1 = connected)")
    plt.show()

if __name__ == "__main__":
    from MultObjPick import Obj
   # from BaselineLearner import AttemptGrips

    voxel_size = 0.015
    half_size = voxel_size / 2
    n_parts = 15
    state_dim = n_parts * 3
    dims = voxel_size * np.ones(3) / 2
    rgb = [[.75, .25, .25]] * n_parts
    obj = Obj(dims, n_parts, rgb)
    obj.GenerateObject(dims, n_parts, [0, 0, 0])
    obj.isMutant = False
    obj.rgb = [tuple(color) for color in obj.rgb]
    colors = [to_hex(rgb) for rgb in obj.rgb]
    visualize_voxel_object(obj.positions,colors)
    a,b = compute_density_field(obj)
    visualize_density(a,b)
    print()