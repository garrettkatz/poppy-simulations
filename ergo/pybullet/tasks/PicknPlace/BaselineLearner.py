import itertools as it
import numpy as np
import matplotlib.pyplot as pt

class BaselineLearner:

    """
    half_grasp: half the grasp distance (in units of voxels) between parallel grippers
    voxel_size: the width of each voxel, in simulator units
    """
    def __init__(self, half_grasp = 1, voxel_size = 0.01):
        self.half_grasp = half_grasp
        self.voxel_size = voxel_size

    """
    Input:
        env: a poppy environment with inverse kinematics API
        voxels[i,j,k]: 1 if object includes voxel at (i,j,k), 0 otherwise
            assumes boundary voxels are all 0
        origin: xyz coordinates of (i,j,k) = (0,0,0) gridpoint (not needed?)
    Output:
        grasp_candidates[n]: nth viable pair of contact points in the voxel grid
    """
    def collect_grasp_candidates(self, env, voxels, origin):
        # pattern is [0, 1, 1, 1, 0], 0 is where the fingertips go
        hg = self.half_grasp
        grasp_pattern = np.ones(2*hg + 3)
        grasp_pattern[[0, -1]] = 0

        grasp_candidates = []
        for (i,j,k) in zip(*np.nonzero(voxels)):
            print(voxels[i-hg-1:i+hg+2, j, k])
            if (voxels[i-hg-1:i+hg+2, j, k] == grasp_pattern).all():
                grasp_candidates.append( ((i-hg-1, j, k), (i+hg+1, j, k)) )
            print(voxels[i, j-hg-1:j+hg+2, k])
            if (voxels[i, j-hg-1:j+hg+2, k] == grasp_pattern).all():
                grasp_candidates.append( ((i, j-hg-1, k), (i, j+hg+1, k)) )
            print(voxels[i, j, k-hg-1:k+hg+2])
            if (voxels[i, j, k-hg-1:k+hg+2] == grasp_pattern).all():
                grasp_candidates.append( ((i, j, k-hg-1), (i, j, k+hg+1)) )

        return grasp_candidates

    """
    Input as above
    Output:
        grasp_candidates as above
        joints[n]: sequence of joint angle waypoints for nth grasp candidate
    """
    def predict_waypoints(self, env, voxels, origin):

        zeros = np.zeros(env.num_joints)

        # relevant joint indices
        fingers = [env.joint_index[f"r_{fm}_tip"] for fm in ("fixed", "moving")]
        neck_index = env.joint_index["head_z"]
        waist_index = 0

        # joints that are free for IK
        free = list(range(env.num_joints))
        free.remove(neck_index)
        free.remove(waist_index)

        # get all grasp candidates
        grasp_candidates = self.collect_grasp_candidates(env, voxels, origin)

        # track successful ones
        viable_grasps, viable_angles = [], []

        # try every grasp
        for n, pair in enumerate(grasp_candidates):
            # try each finger at each contact point
            for both in (pair, pair[::-1]):

                # convert to cartesian coordinates
                targets = [self.voxel_size * np.array(p) + origin for p in both]
                
                # try to reach
                angles, out_of_range, error = env.partial_ik(fingers, targets, zeros, free)
                if out_of_range: continue
                if error > self.voxel_size / 10: continue

                viable_grasps.append(both)
                viable_angles.append(angles)

        return viable_grasps, viable_angles

    """
    Input:
        env, voxels: same as predict_waypoints
        grasp_candidates, waypoints: output of previous call to predict_waypoints
        reward: reward returned by the environment after attempting the waypoints
    """
    def learn(self, env, voxels, grasp_candidates, waypoints, reward):
        pass

if __name__ == "__main__":

    # grid_size = 10
    # num_voxels = 30

    # # random voxel grid generation
    # nonzeros = np.empty((num_voxels, 3), dtype=int)
    # nonzeros[0,:] = grid_size//2
    # for v in range(1, num_voxels):
    #     n = np.random.randint(v)
    #     nonzeros[v] = nonzeros[n]
    #     nonzeros[v, np.random.randint(3)] += np.sign(np.random.randn())
    # nonzeros = nonzeros[(0 < nonzeros).all(axis=1) & (nonzeros < grid_size-1).all(axis=1)]
    # voxels = np.zeros((grid_size,)*3)
    # voxels[tuple(nonzeros.T)] = 1

    # # learner API
    # env, origin = None, None
    # learner = BaselineLearner()
    # grasp_candidates, waypoints  = learner.predict_waypoints(env, voxels, origin)

    # # TBD:
    # # reward = env.run_trajectory(waypoints)
    # # learner.learn(env, voxels, grasp_candidates, waypoints, reward)

    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.voxels(voxels.astype(bool))
    # ax.scatter(*zip(*(grasp_candidates + 0.5)), color='red')
    # pt.show()

    import MultObjPick
    import pybullet as pb

    half_grasp = 0 # grasp points at +/- (half_grasp+1)
    voxel_size = 0.01 # dimension of each voxel

    learner = BaselineLearner(half_grasp, voxel_size)
    
    Experiment_env = MultObjPick.experiment()
    Experiment_env.CreateScene()
    
    dims = voxel_size * np.ones(3)
    n_parts = 6
    rgb = [(.75, .25, .25)] * n_parts
    Experiment_obj = MultObjPick.Obj(dims,n_parts,rgb)
    Experiment_obj.GenerateObject(dims,n_parts,[0,0,0])
    Experiment_env.Spawn_Object(Experiment_obj)

    # replace this block with Actual_Voxels?
    vpos = np.array(Experiment_obj.positions)
    vijk = (vpos / (2*voxel_size)).round().astype(int)
    vijk -= vijk.min()
    vijk += half_grasp + 1 # padding
    grid_size = vijk.max() + half_grasp +2 # padding
    voxels = np.zeros((grid_size,)*3)
    for (i,j,k) in vijk: voxels[i,j,k] = 1

    # # plot the voxel grid
    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.voxels(voxels.astype(bool))
    # pt.show()

    # get all the grasp candidates
    origin = Experiment_obj.basePosition
    grasp_candidates = learner.collect_grasp_candidates(Experiment_env.env, voxels, origin)

    # convert one grasp candidate to simulator coordinates
    cand = np.array(grasp_candidates.pop(), dtype=float)
    cand -= half_grasp + 1 # subtract padding
    cand += 0.5 # add half-voxel for centering
    cand = voxel_size * cand + origin # transform to simulator coordinates

    # need to debug
    pb.addUserDebugPoints(cand, [[0.,1.,0.]]*2, 10.0)

    input('.')

    
    
