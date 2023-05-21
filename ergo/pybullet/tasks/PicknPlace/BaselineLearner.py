import sys, os
import itertools as it
import numpy as np
import matplotlib.pyplot as pt

sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents

class BaselineLearner:

    """
    grasp_width: max distance between grippers when grasping, in voxel units
    voxel_size: the width of each voxel, in simulator units
    """
    def __init__(self, grasp_width, voxel_size):
        self.grasp_width = grasp_width
        self.voxel_size = voxel_size

    """
    Input:
        voxels[i,j,k]: 1 if object includes voxel at (i,j,k), 0 otherwise
            assumes boundary voxels are all 0
    Output:
        grasp_candidates[n]: nth viable pair of contact points in the voxel grid coordinates
            (0,0,0) is low corner of lowest voxel in all three dims
    """
    def collect_grasp_candidates(self, voxels):
        grasp_candidates = []

        # try all grasp widths up to max
        for gw in range(1, self.grasp_width+1):

            # pattern is like [0, 1, 0], 0 is where the fingertips go
            grasp_pattern = np.ones(gw + 2)
            grasp_pattern[[0, -1]] = 0
    
            for (i,j,k) in zip(*np.nonzero(voxels)):
                if (voxels[i-1:i+gw+1, j, k] == grasp_pattern).all():
                    grasp_candidates.append( ((i, j+.5, k+.5), (i+gw, j+.5, k+.5)) )
                if (voxels[i, j-1:j+gw+1, k] == grasp_pattern).all():
                    grasp_candidates.append( ((i+.5, j, k+.5), (i+.5, j+gw, k+.5)) )
                if (voxels[i, j, k-1:k+gw+1] == grasp_pattern).all():
                    grasp_candidates.append( ((i+.5, j+.5, k), (i+.5, j+.5, k+gw)) )

        return grasp_candidates

    """
    Create joint trajectory to pick up an object at specified grip_points
    input: grip_points[i]: xyz coordinates for ith finger
    output: trajectory[n]: joint angles for nth waypoint
    """
    def get_pick_trajectory(self, env, grip_points):

        # choose arm based on x-coordinate of grip points
        if grip_points.mean(axis=0)[0] > 0:
            arm, other_arm = "lr"
        else:
            arm, other_arm = "rl"

        # links with specified IK targets
        links = [
            env.joint_index[f"{arm}_moving_tip"],
            env.joint_index[f"{arm}_fixed_tip"],
            env.joint_index[f"{arm}_fixed_knuckle"], # for top-down approach
        ]

        # joints that are free for IK
        free = list(range(env.num_joints))
        # links with targets are not considered free
        for idx in links: free.remove(idx)
        # keep reasonable posture
        free.remove(env.joint_index["head_z"]) # neck
        free.remove(0) # waist

        # get waypoint targets
        grip_centers = grip_points.mean(axis=0, keepdims=True)
        open_down = grip_centers + 1.5 * (grip_points - grip_centers)
        closed_down = grip_centers + 0.9 * (grip_points - grip_centers)
        closed_up = closed_down + np.array([[0, 0, 0.1]])
        open_up = open_down + np.array([[0, 0, 0.1]])
        waypoints = (open_up, open_down, closed_down, closed_up)

        # try each finger at each contact point
        start_angles = env.get_position()
        max_error = [0]*2
        trajectories = []
        for i, (fixed, moving) in enumerate(([0, 1], [1, 0])):

            # IK on each waypoint
            env.set_position(start_angles)
            angles = {-1: start_angles}
            for w in range(len(waypoints)):

                # targets for current assignment of fingers to contact points
                targets = waypoints[w][[moving, fixed, fixed], :]
                targets[2,2] += .05 # xyz origin offset of fixed_knuckle in urdf
            
                # try to reach
                angles[w], out_of_range, error = env.partial_ik(links, targets, angles[w-1], free, num_iters=3000)
                max_error[i] = max(max_error[i], error)

                # don't change other arm angles
                for joint_name in ("shoulder_x", "shoulder_y", "arm_z", "elbow_y"):
                    idx = env.joint_index[f"{other_arm}_{joint_name}"]
                    # angles[w][idx] = 0
                    angles[w][idx] = start_angles[idx]

                # env.set_position(angles[w])
                # print(max_error[i])
                # input('..')

            trajectory = [angles[w] for w in range(len(waypoints))]
            trajectories.append(trajectory)

            env.set_position(start_angles)

        # choose trajectory with lower max error
        choice = np.argmin(max_error)
        trajectory = trajectories[choice]

        return trajectory

        # # order best (least error) to worst


        # # track successful trajectories and their errors
        # trajectories = []
        # max_errors = []

        # # error tolerance for IK solution
        # errtol = self.voxel_size * 0.1

        # # try every grasp
        # for n in range(len(grasp_candidates)):
        #     print(f"trying {n} of {len(grasp_candidates)}, {len(trajectories)} trajectories so far")
        #     # try each finger at each contact point
        #     for fixed, moving in ([0, 1], [1, 0]):

        #         # IK on each waypoint
        #         env.set_position(rest)
        #         angles = {-1: rest}
        #         max_error = 0
        #         for w in range(len(waypoints)):

        #             targets = waypoints[w][n][[moving, fixed, fixed], :]
        #             targets[2,2] += .05 # xyz origin offset of fixed_knuckle in urdf
                
        #             # try to reach
        #             angles[w], out_of_range, error = env.partial_ik(links, targets, angles[w-1], free, num_iters=3000)
        #             # if out_of_range: break
        #             max_error = max(max_error, error)

        #         angles = [angles[w] for w in range(len(waypoints))]

        #         # env.set_position(angles[1])
        #         # print(out_of_range)
        #         # print(max_error, errtol)
        #         # input('..')

        #         # save trajectories
        #         trajectories.append(angles)
        #         max_errors.append(max_error)

        # # order best (least error) to worst
        # sorter = np.argsort(max_errors)
        # trajectories = [trajectories[s] for s in sorter]
        # max_errors = [max_errors[s] for s in sorter]

        # return trajectories

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

    grasp_width = 1 # distance between grippers in voxel units
    voxel_size = 0.03 # dimension of each voxel
    table_height = table_position()[2] + table_half_extents()[2] # z coordinate of table surface

    learner = BaselineLearner(grasp_width, voxel_size)
    
    exp = MultObjPick.experiment()
    exp.CreateScene()
    env = exp.env

    # better view of tabletop
    pb.resetDebugVisualizerCamera(
        cameraDistance = 1.4,
        cameraYaw = -1.2,
        cameraPitch = -39.0,
        cameraTargetPosition = (0., 0., 0.),
    )
    
    dims = voxel_size * np.ones(3) / 2 # dims are actually half extents
    n_parts = 6
    rgb = [(.75, .25, .25)] * n_parts
    obj = MultObjPick.Obj(dims,n_parts,rgb)
    obj.GenerateObject(dims,n_parts,[0,0,0])
    obj_id = exp.Spawn_Object(obj)

    # convert positions to voxel grid
    voxel_offsets = np.array(obj.positions) # offset from base pos to voxel centers
    voxel_corner = voxel_offsets.min(axis=0) - voxel_size/2 # lower corner of voxels
    vijk = ((voxel_offsets - voxel_offsets.min(axis=0)) / voxel_size).round().astype(int)
    vijk += 1 # lower padding
    grid_size = vijk.max(axis=0) + grasp_width + 2 # upper padding
    voxels = np.zeros(tuple(grid_size))
    for (i,j,k) in vijk: voxels[i,j,k] = 1

    # get candidates and convert back to simulator units
    # coords[n,i,:]: xyz coordinates if ith contact point in nth grasp
    cands = np.array(learner.collect_grasp_candidates(voxels))
    coords = (cands - 1) * voxel_size + voxel_corner # undo padding, rescale and recenter

    # # transform to object location
    # coords += np.array(obj.basePosition)

    # # visualize grasp points in simulator
    # pb.addUserDebugPoints(coords[0], [[0.,1.,0.]]*2, 25.0)

    # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
    orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
    exp.env.settle(exp.env.get_position(), seconds=2)
    rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

    # abort if object fell off of table
    if rest_pos[2] < table_height: sys.exit(0)

    # transform grasp coordinates to object pose
    M = pb.getMatrixFromQuaternion(rest_orn) # orientation of rest object in world coordinates
    M = np.array(M).reshape(3,3)
    rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

    # # visualize grasp points in simulator
    # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

    # select highest grasp coordinates (heuristic to avoid object-gripper collision)
    hi = rest_coords.mean(axis=1)[:,2].argmax()
    grip_points = rest_coords[hi]
    trajectory = learner.get_pick_trajectory(exp.env, grip_points)

    # # reposition objected flush on table
    # base_pos = np.array(obj.basePosition)
    # voxel_pos = np.array(obj.positions)
    # voxel_centers = voxel_pos + base_pos
    # table_height = table_position()[2] + table_half_extents()[2]
    # z_offset = (voxel_centers[:,2].min() - voxel_size/2) - table_height - 0.001 # .001 to avoid initial collision
    # base_pos[2] -= z_offset
    # voxel_centers[:,2] -= z_offset
    # voxel_origin = voxel_centers.min(axis=0) - voxel_size/2
    # pb.resetBasePositionAndOrientation(obj_id, base_pos, (0.0, 0.0, 0.0, 1))
    # base_pos, base_orn = pb.getBasePositionAndOrientation(obj_id)
    # # pb.addUserDebugPoints(voxel_centers, [[0.,1.,0.]]*len(voxel_centers), 25.0)

    # # get all the grasp candidates and choose one of them
    # grasp_candidates = learner.collect_grasp_candidates(exp.env, voxels, voxel_origin)
    # cand = np.array(grasp_candidates.pop(), dtype=float)

    # # convert candidate to simulator units and coordinates
    # coords = (cand - 1) * voxel_size # subtract lower padding and convert units
    # coords += voxel_origin # transform to object position

    # # visualize grasp points in simulator
    # pb.addUserDebugPoints(coords, [[0.,1.,0.]]*2, 10.0)
    # pb.addUserDebugPoints([voxel_origin], [[0.,0.,1.]], 25.0)

    # trajectories = learner.predict_waypoints(exp.env, voxels, voxel_origin)
    # print(len(trajectories))

    # visualize grasp points in the voxel grid
    ax = pt.gcf().add_subplot(projection='3d')
    ax.voxels(voxels.astype(bool), alpha=0.5)
    pt.plot(*cands[0].T, marker='o', color='red')
    pt.show()

    # try best trajectory
    for angles in trajectory:
        exp.env.goto_position(angles, duration = 2)
        exp.env.goto_position(angles, duration = .1) # in case it needs a little more time to converge
        # input('.')

    # input('.')

    
