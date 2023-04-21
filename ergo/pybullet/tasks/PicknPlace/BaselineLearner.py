import itertools as it
import numpy as np
import matplotlib.pyplot as pt

class BaselineLearner:

    def __init__(self):
        pass

    """
    Input:
        env: a poppy environment with inverse kinematics API
        voxels[i,j,k]: 1 if object includes voxel at (i,j,k), 0 otherwise
            assumes boundary voxels are all 0
        origin: xyz coordinates of (i,j,k) = (0,0,0) gridpoint (not needed?)
    Output:
        grasp points in the voxel grid for each finger tip
        sequence of joint angle waypoints for grasping motion
    """
    def predict_waypoints(self, env, voxels, origin):
        half_grasp = 1

        grasp_pattern = np.ones(2*half_grasp + 3)
        grasp_pattern[[0, -1]] = 0

        grasppoints = None
        for (i,j,k) in zip(*np.nonzero(voxels)):
            if (voxels[i-half_grasp-1:i+half_grasp+2, j, k] == grasp_pattern).all():
                grasppoints = ((i-half_grasp-1, j, k), (i+half_grasp+1, j, k))
            if (voxels[i, j-half_grasp-1:j+half_grasp+2, k] == grasp_pattern).all():
                grasppoints = ((i, j-half_grasp-1, k), (i, j+half_grasp+1, k))
            if (voxels[i, j, k-half_grasp-1:k+half_grasp+2] == grasp_pattern).all():
                grasppoints = ((i, j, k-half_grasp-1), (i, j, k+half_grasp+1))
            if grasppoints is not None: break

        return np.array(grasppoints), None

    """
    Input:
        env, voxels: same as predict_waypoints
        grasppoints, waypoints: output of previous call to predict_waypoints
        reward: reward returned by the environment after attempting the waypoints
    """
    def learn(self, env, voxels, grasppoints, waypoints, reward):
        pass

if __name__ == "__main__":

    grid_size = 10
    num_voxels = 30

    # random voxel grid generation
    nonzeros = np.empty((num_voxels, 3), dtype=int)
    nonzeros[0,:] = grid_size//2
    for v in range(1, num_voxels):
        n = np.random.randint(v)
        nonzeros[v] = nonzeros[n]
        nonzeros[v, np.random.randint(3)] += np.sign(np.random.randn())
    nonzeros = nonzeros[(0 < nonzeros).all(axis=1) & (nonzeros < grid_size-1).all(axis=1)]
    voxels = np.zeros((grid_size,)*3)
    voxels[tuple(nonzeros.T)] = 1

    # learner API
    env, origin = None, None
    learner = BaselineLearner()
    grasppoints, waypoints  = learner.predict_waypoints(env, voxels, origin)

    # TBD:
    # reward = env.run_trajectory(waypoints)
    # learner.learn(env, voxels, grasppoints, waypoints, reward)

    ax = pt.gcf().add_subplot(projection='3d')
    ax.voxels(voxels.astype(bool))
    ax.scatter(*zip(*(grasppoints + 0.5)), color='red')
    pt.show()
    
    
