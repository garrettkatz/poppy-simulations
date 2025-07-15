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
    Get padded voxel representation of object
    """
    def object_to_voxels(self, obj):
        #self.voxel_size = 0.015
        # get offsets from object base position to voxel centers
        voxel_offsets = np.array(obj.positions)

        # convert simulation units to grid (i,j,k) indices
        vijk = ((voxel_offsets - voxel_offsets.min(axis=0)) / self.voxel_size).round().astype(int)

        # pad grid to check that voxels next to contact points are empty
        pad = 1
        vijk += pad # lower padding
        grid_size = vijk.max(axis=0) + self.grasp_width + pad + 1 # upper padding

        # fill in voxel grid
        voxels = np.zeros(tuple(grid_size))
        for (i,j,k) in vijk: voxels[i,j,k] = 1

        # also get coordinates for lower grid corner corner to convert back later
        voxel_corner = voxel_offsets.min(axis=0) - self.voxel_size/2

        return voxels, voxel_corner

    """
    Input:
        voxels[i,j,k]: 1 if object includes voxel at (i,j,k), 0 otherwise
            assumes padding of 1, meaning boundary voxels are all 0
    Output:
        cands[n,i,:]: voxel grid coordinates of ith contact point in nth grasp, i in (0, 1)
    """
    def collect_grasp_candidates(self, voxels):
        candidates = []

        # try all grasp widths up to max
        for gw in range(1, self.grasp_width+1):

            # pattern is like [0, 1, 0], 0 is where the fingertips go
            grasp_pattern = np.ones(gw + 2)
            grasp_pattern[[0, -1]] = 0
    
            for (i,j,k) in zip(*np.nonzero(voxels)):
                if (voxels[i-1:i+gw+1, j, k] == grasp_pattern).all():
                    candidates.append( ((i, j+.5, k+.5), (i+gw, j+.5, k+.5)) )
                if (voxels[i, j-1:j+gw+1, k] == grasp_pattern).all():
                    candidates.append( ((i+.5, j, k+.5), (i+.5, j+gw, k+.5)) )
                if (voxels[i, j, k-1:k+gw+1] == grasp_pattern).all():
                    candidates.append( ((i+.5, j+.5, k), (i+.5, j+.5, k+gw)) )

        return np.array(candidates)

    """
    Convert grip coordinates from voxel to simulation units
        candidates: as returned by collect_grasp_candidates
        voxel_corner: lower coordinates of voxel grid for conversion back to simulator units
    Output:
        coords[n,i,:]: simulator coordinates of ith contact point in nth grasp, i in (0, 1)
    """
    def voxel_to_sim_coords(self, candidates, voxel_corner):
        coords = (candidates - 1) # undo padding
        coords = coords * self.voxel_size + voxel_corner # rescale and recenter
        return coords

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
                if angles[w][env.joint_index[f"{arm}_gripper"]] <0.5:
                    continue
                max_error[i] = max(max_error[i], error)
                #max_error.append(max(max_error[i], error))

                # don't change other arm angles
                for joint_name in ("shoulder_x", "shoulder_y", "arm_z", "elbow_y"):
                    idx = env.joint_index[f"{other_arm}_{joint_name}"]
                    # angles[w][idx] = 0
                    angles[w][idx] = start_angles[idx]
                
                # env.set_position(angles[w])
                # print(max_error[i])
                # input('..')
            #angle[w][x] < 0 , continue
            trajectory = [angles[w] for w in range(len(waypoints))]
           # if trajectory[len(trajectory) - 1][env.joint_index[f"{arm}_gripper"]] > 0.5:
          #      continue
           # if trajectory[len(trajectory) - 1][env.joint_index[f"{other_arm}_gripper"]] > 0.5:
           #     continue
            trajectories.append(trajectory)
            env.set_position(start_angles)

        # choose trajectory with lower max error
        choice = np.argmin(max_error)

        trajectory = trajectories[choice]

        return trajectory


    def get_pick_trajectory_variation(self, env, grip_points):

            # choose arm based on x-coordinate of grip points
            if grip_points.mean(axis=0)[0] > 0:
                arm, other_arm = "lr"
            else:
                arm, other_arm = "rl"

            # links with specified IK targets
            links = [
                env.joint_index[f"{arm}_moving_tip"],
                env.joint_index[f"{arm}_fixed_tip"],
                env.joint_index[f"{arm}_fixed_knuckle"],  # for top-down approach
            ]

            # joints that are free for IK
            free = list(range(env.num_joints))
            # links with targets are not considered free
            for idx in links: free.remove(idx)
            # keep reasonable posture
            free.remove(env.joint_index["head_z"])  # neck
            free.remove(0)  # waist

            # get waypoint targets
            grip_centers = grip_points.mean(axis=0, keepdims=True)
            open_down = grip_centers + 1.5 * (grip_points - grip_centers)
            closed_down = grip_centers + 0.9 * (grip_points - grip_centers)
            closed_up = closed_down + np.array([[0, 0, 0.1]])
            open_up = open_down + np.array([[0, 0, 0.1]])
            waypoints = (open_up, open_down, closed_down, closed_up)

            # try each finger at each contact point
            start_angles = env.get_position()
            max_error = [0] * 2
            trajectories = []
            for i, (fixed, moving) in enumerate(([0, 1], [1, 0])):

                # IK on each waypoint
                env.set_position(start_angles)
                angles = {-1: start_angles}
                for w in range(len(waypoints)):

                    # targets for current assignment of fingers to contact points
                    targets = waypoints[w][[moving, fixed, fixed], :]
                    targets[2, 2] += .05  # xyz origin offset of fixed_knuckle in urdf

                    # try to reach
                    angles[w], out_of_range, error = env.partial_ik(links, targets, angles[w - 1], free, num_iters=3000)
                    if angles[w][env.joint_index[f"{arm}_gripper"]] < 0.5:
                        continue
                    max_error[i] = max(max_error[i], error)
                    # max_error.append(max(max_error[i], error))

                    # don't change other arm angles
                    for joint_name in ("shoulder_x", "shoulder_y", "arm_z", "elbow_y"):
                        idx = env.joint_index[f"{other_arm}_{joint_name}"]
                        # angles[w][idx] = 0
                        angles[w][idx] = start_angles[idx]

                    # env.set_position(angles[w])
                    # print(max_error[i])
                    # input('..')
                # angle[w][x] < 0 , continue
                trajectory = [angles[w] for w in range(len(waypoints))]
                # if trajectory[len(trajectory) - 1][env.joint_index[f"{arm}_gripper"]] > 0.5:
                #      continue
                # if trajectory[len(trajectory) - 1][env.joint_index[f"{other_arm}_gripper"]] > 0.5:
                #     continue
                trajectories.append(trajectory)
                env.set_position(start_angles)

            # choose trajectory with lower max error
            choice = np.argmin(max_error)

            trajectory = trajectories[choice]

            return trajectory,arm

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



    def baseline_learner_experiment():
        #Old main written by Prof. Katz as a sanity check
        import MultObjPick
        import pybullet as pb

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
        obj_id = exp.Spawn_Object(obj)

        # get voxel grid for object
        voxels, voxel_corner = learner.object_to_voxels(obj)

        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)

        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)

        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height: sys.exit(0)

        # transform grasp coordinates to object pose
        M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
        M = np.array(M).reshape(3, 3)
        rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

        # # visualize grasp points in simulator
        # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

        # select highest grasp coordinates
        # (heuristic to avoid object-gripper collision in top-down grasps)
        hi = rest_coords.mean(axis=1)[:, 2].argmax()
        grip_points = rest_coords[hi]
        trajectory = learner.get_pick_trajectory(exp.env, grip_points)

        # visualize grasp points in the voxel grid
        ax = pt.gcf().add_subplot(projection='3d')
        ax.voxels(voxels.astype(bool), alpha=0.5)
        pt.plot(*cands[0].T, marker='o', color='red')
        pt.show()

        # try best trajectory
        for angles in trajectory:
            exp.env.goto_position(angles, duration=2)
            exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
            # input('.')
        picked_pos,_ = pb.getBasePositionAndOrientation(obj_id)
        if picked_pos[2] > rest_pos[2]:
            print("pick success!")

        # input('.')
    def Experiment1_MultipleRandomObjects_OneGrip(self):
        #Experiment details:
        #One set of grip points selected from a list of candidate grips per object.
        import MultObjPick
        import pybullet as pb

        grasp_width = 1  # distance between grippers in voxel units
        voxel_size = 0.03  # dimension of each voxel
        table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

        learner = BaselineLearner(grasp_width, voxel_size)
        Num_success = 0
        Num_Grips_attempted = 0

        Result = []
        for iter in range(30):
            exp = MultObjPick.experiment()
            exp.CreateScene()
            env = exp.env

            pb.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=-1.2,
                cameraPitch=-39.0,
                cameraTargetPosition=(0., 0., 0.),
            )
            # better view of tabletop

            dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
            n_parts = 6
            rgb = [(.75, .25, .25)] * n_parts
            obj = MultObjPick.Obj(dims, n_parts, rgb)
            obj.GenerateObject(dims, n_parts, [0, 0, 0])
            obj_id = exp.Spawn_Object(obj)

            # get voxel grid for object
            voxels, voxel_corner = learner.object_to_voxels(obj)

            # get candidate grasp points
            cands = learner.collect_grasp_candidates(voxels)

            # convert back to simulator units
            coords = learner.voxel_to_sim_coords(cands, voxel_corner)

            # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
            orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
            exp.env.settle(exp.env.get_position(), seconds=3)
            rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

            # abort if object fell off of table
            if rest_pos[2] < table_height:
                pb.removeBody(obj_id)
                exp.env.close()
                continue
            # sys.exit(0)

            # transform grasp coordinates to object pose
            M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
            M = np.array(M).reshape(3, 3)
            rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

            # # visualize grasp points in simulator
            # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

            # select highest grasp coordinates
            # (heuristic to avoid object-gripper collision in top-down grasps)
            hi = rest_coords.mean(axis=1)[:, 2].argmax()
            grip_points = rest_coords[hi]
            trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            # visualize grasp points in the voxel grid
            #ax = pt.gcf().add_subplot(projection='3d')
            #ax.voxels(voxels.astype(bool), alpha=0.5)
            #pt.plot(*cands[0].T, marker='o', color='red')
            #pt.show()
            # try best trajectory
            for angles in trajectory:
                exp.env.goto_position(angles, duration=2)
                exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
                # input('.')
            picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
            Num_Grips_attempted = Num_Grips_attempted + 1
            if picked_pos[2] > rest_pos[2]:
                print("pick success! -- ", obj_id)
                Num_success = Num_success + 1
            Result.append((picked_pos[2]-rest_pos[2])*10)
            pb.removeBody(obj_id)
            exp.env.close()
        print("\nNum of grip attempt:", Num_Grips_attempted)
        print("\n Num of successful picks", Num_success)
        import matplotlib.pyplot as plt
        plt.plot(Result)
        plt.ylabel("Z-axis difference")
        plt.show()


def Experiment2_MultipleRandomObjects_AllCandidateGrips():
    #Experiment details:
    #All candidate grips are attempted per object.
    import MultObjPick
    import pybullet as pb

    grasp_width = 1  # distance between grippers in voxel units
    voxel_size = 0.03  # dimension of each voxel
    table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

    learner = BaselineLearner(grasp_width, voxel_size)
    Num_success = 0
    Num_Grips_attempted = 0
    exp = MultObjPick.experiment()
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=-1.2,
        cameraPitch=-39.0,
        cameraTargetPosition=(0., 0., 0.),
    )
    Result = []
    for iter in range(100):
        exp.CreateScene()
        env = exp.env
        dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
        n_parts = 6
        rgb = [(.75, .25, .25)] * n_parts
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        obj_id = exp.Spawn_Object(obj)

        # get voxel grid for object
        voxels, voxel_corner = learner.object_to_voxels(obj)

        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)

        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)

        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=2)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height:
            pb.removeBody(obj_id)
            continue
        # sys.exit(0)

        # transform grasp coordinates to object pose
        M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
        M = np.array(M).reshape(3, 3)
        rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

        # # visualize grasp points in simulator
        # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

        # select highest grasp coordinates
        # (heuristic to avoid object-gripper collision in top-down grasps)
        hi = rest_coords.mean(axis=1)[:, 2].argmax()
        interm_result = 0
        for i in range(len(rest_coords)-1): # choosing all candidates
            if i > 0: # respawn object after an attempt.
                obj_id = exp.Spawn_Object(obj)
                orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
                exp.env.settle(exp.env.get_position(), seconds=2)
                rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

                # abort if object fell off of table
                if rest_pos[2] < table_height:
                    pb.removeBody(obj_id)
                    continue
                # sys.exit(0)

                # transform grasp coordinates to object pose
                M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
                M = np.array(M).reshape(3, 3)
                rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
            grip_points = rest_coords[i]
            trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            # visualize grasp points in the voxel grid
            ax = pt.gcf().add_subplot(projection='3d')
            ax.voxels(voxels.astype(bool), alpha=0.5)
            pt.plot(*cands[i].T, marker='o', color='red')
            pt.show()
            # try best trajectory
            for angles in trajectory:
                exp.env.goto_position(angles, duration=2)
                exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
                # input('.')
            picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
            Num_Grips_attempted = Num_Grips_attempted + 1
            if picked_pos[2] > rest_pos[2]:
                print("pick success! -- ", obj_id)
                Num_success = Num_success + 1
            interm_result = interm_result +(picked_pos[2] - rest_pos[2])



            pb.removeBody(obj_id)
        Result.append(interm_result / len(rest_coords))
        print("\nNum of grip attempt:", Num_Grips_attempted)
        print("\n Num of successful picks", Num_success)
        import matplotlib.pyplot as plt
        plt.plot(Result)
        plt.ylabel("Avg Z-axis difference")
        plt.show()
    def PerformExperimentAllgrips(exp,learner,grip_points):
        import MultObjPick
        import pybullet as pb


def Experiment3_AdversarialBaseline_oneObjectMutant_OneGrip():
    import MultObjPick
    import pybullet as pb
    attemptCounter = 0
    grasp_width = 1  # distance between grippers in voxel units
    voxel_size = 0.03  # dimension of each voxel
    table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface

    learner = BaselineLearner(grasp_width, voxel_size)
    Num_success = 0
    Num_Grips_attempted = 0

    Result = []
    for iter in range(2):
        exp = MultObjPick.experiment()
        exp.CreateScene()
        env = exp.env
        Obj_result = []
        Obj_result.clear()
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=-1.2,
            cameraPitch=-39.0,
            cameraTargetPosition=(0., 0., 0.),
        )
        # better view of tabletop

        dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
        n_parts = 6
        rgb = [(.75, .25, .25)] * n_parts
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        obj_id = exp.Spawn_Object(obj)
        # Mutant = obj.MutateObject()
        mutants = obj.Multiple_MutateObject()
        voxels, voxel_corner = learner.object_to_voxels(obj)

        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)

        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)

        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height:
            pb.removeBody(obj_id)
            exp.env.close()
            continue
        # sys.exit(0)

        # transform grasp coordinates to object pose
        M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
        M = np.array(M).reshape(3, 3)
        rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

        # # visualize grasp points in simulator
        # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

        # select highest grasp coordinates
        # (heuristic to avoid object-gripper collision in top-down grasps)
        hi = rest_coords.mean(axis=1)[:, 2].argmax()
        grip_points = rest_coords[hi]
        trajectory = learner.get_pick_trajectory(exp.env, grip_points)

        # visualize grasp points in the voxel grid
        # ax = pt.gcf().add_subplot(projection='3d')
        # ax.voxels(voxels.astype(bool), alpha=0.5)
        # pt.plot(*cands[0].T, marker='o', color='red')
        # pt.show()
        # try best trajectory
        for angles in trajectory:
            exp.env.goto_position(angles, duration=2)
            exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
            # input('.')
        picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
        Num_Grips_attempted = Num_Grips_attempted + 1
        if picked_pos[2] > rest_pos[2]:
            print("\nSuccess , Total Attempt number ",Num_Grips_attempted )
            Num_success = Num_success + 1
        Obj_result.append((picked_pos[2] - rest_pos[2]) * 10)
        pb.removeBody(obj_id)
        exp.reset_robot()
        for mut in mutants:
            # obj = MultObjPick.Obj(mut.dims, n_parts, rgb)
            # obj.GenerateObject(dims, n_parts, [0, 0, 0])
            exp.reset_robot()
            obj_id = exp.Spawn_Object(mut)
            voxels, voxel_corner = learner.object_to_voxels(mut)
            # get candidate grasp points
            cands = learner.collect_grasp_candidates(voxels)
            # convert back to simulator units
            coords = learner.voxel_to_sim_coords(cands, voxel_corner)
            orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
            exp.env.settle(exp.env.get_position(),
                           seconds=3)  # wait for object to settle since its dropped from a small height
            rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)
            if rest_pos[2] < table_height:  # object falls/tumbles off , ignore this case
                pb.removeBody(obj_id)
                exp.env.close()
                continue
                # transform grasp coordinates to object pose
                # transform grasp coordinates to object pose
            M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
            M = np.array(M).reshape(3, 3)
            rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

            # # visualize grasp points in simulator
            # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

            # select highest grasp coordinates
            # (heuristic to avoid object-gripper collision in top-down grasps)
            hi = rest_coords.mean(axis=1)[:, 2].argmax()
            grip_points = rest_coords[hi]
            trajectory = learner.get_pick_trajectory(exp.env, grip_points)
            for angles in trajectory:
                exp.env.goto_position(angles, duration=2)
                exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
                # input('.')
            picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
            Num_Grips_attempted = Num_Grips_attempted + 1
            if picked_pos[2] > rest_pos[2]:
                print("\nSuccess , Total Attempt number ",Num_Grips_attempted )
                Num_success = Num_success + 1
            Obj_result.append((picked_pos[2] - rest_pos[2]) * 10)
            pb.removeBody(obj_id)
        Result.append(Obj_result)
        exp.env.close()

    print("\nNum of grip attempt:", Num_Grips_attempted)
    print("\n Num of successful picks", Num_success)
    import matplotlib.pyplot as plt
    for r in Result:
        plt.plot(r)
    plt.ylabel("Z-axis difference")
    plt.xlabel("Mutations")
    plt.show()


def AttemptGrips(obj,gen):
    import MultObjPick
    import pybullet as pb

    grasp_width = 1  # distance between grippers in voxel units
    voxel_size = 0.015  # dimension of each voxel
    table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface
    learner = BaselineLearner(grasp_width, voxel_size)
    Num_success = 0
    Num_Grips_attempted = 0
    Result = []
    num_objects = 1

    exp = MultObjPick.experiment()
    exp.CreateScene()
    env = exp.env

    pb.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=-1.2,
        cameraPitch=-39.0,
        cameraTargetPosition=(0., 0., 0.),
    )
    obj_id = exp.Spawn_Object(obj)
    # Mutant = obj.MutateObject()
    voxels, voxel_corner = learner.object_to_voxels(obj)
    # get candidate grasp points
    cands = learner.collect_grasp_candidates(voxels)
    # convert back to simulator units
    coords = learner.voxel_to_sim_coords(cands, voxel_corner)
    # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
    orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
    exp.env.settle(exp.env.get_position(), seconds=3)
    rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

    # abort if object fell off of table
    if rest_pos[2] < table_height:
        pb.removeBody(obj_id)
        exp.env.close()
        return 0,-1
    # sys.exit(0)
    M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
    M = np.array(M).reshape(3, 3)
    rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
    #interm_result = []
    res = []
    num_grips_success = 0

    for i in range(len(rest_coords) - 1):  # choosing all candidates
        if i > 0:  # respawn object after an attempt.
            exp.reset_robot()
            obj_id = exp.Spawn_Object(obj)
            orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
            exp.env.settle(exp.env.get_position(), seconds=3)
            rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

            # abort if object fell off of table
            if rest_pos[2] < table_height:
                pb.removeBody(obj_id)
                continue
            # sys.exit(0)

            # transform grasp coordinates to object pose
            M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
            M = np.array(M).reshape(3, 3)
            rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
        grip_points = rest_coords[i]
        trajectory = learner.get_pick_trajectory(exp.env, grip_points)

        # try best trajectory
        for i in range(len(trajectory)):
            #trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            exp.env.goto_position(trajectory[i], duration=2)
            exp.env.goto_position(trajectory[i], duration=.1)  # in case it needs a little more time to converge
            #trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            # input('.')
        picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
        height = (picked_pos[2] - rest_pos[2])
        if ((picked_pos[2]-rest_pos[2])>0.004):
            num_grips_success = num_grips_success+1
            print("S")
        elif ((picked_pos[2]-rest_pos[2])<=0.004):
            height = 0
            print("F")
        res.append(height)
        pb.removeBody(obj_id)
    max_res = max(res)
    avg_res = (sum(res)/len(res))*10
    #pb.disconnect()
    env.close()
    if num_grips_success>0 and gen==0:
        print("dumping to file , good obj")
        with open(f'mutantGen_{gen}_OBJ_details.pickle', 'ab+') as handle:
            pickle.dump(obj.positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    if gen>0:
        print("dumping to file")
        with open(f'mutantGen_{gen}_OBJ_details.pickle', 'ab+') as handle:
            pickle.dump(obj.positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    return max_res , num_grips_success

def BigTest():

    import MultObjPick
    import pybullet as pb

    grasp_width = 1  # distance between grippers in voxel units
    voxel_size = 0.03  # dimension of each voxel
    table_height = table_position()[2] + table_half_extents()[2]  # z coordinate of table surface
    obj_result = []
    learner = BaselineLearner(grasp_width, voxel_size)
    Num_success = 0
    Num_Grips_attempted = 0

    Result = []
    num_objects= 1
    for iter in range(num_objects):
        exp = MultObjPick.experiment()
        exp.CreateScene()
        env = exp.env
        Obj_result = []
        Obj_result.clear()
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=-1.2,
            cameraPitch=-39.0,
            cameraTargetPosition=(0., 0., 0.),
        )
        # better view of tabletop

        dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
        n_parts = 20
        rgb = [(.75, .25, .25)] * n_parts
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        obj_id = exp.Spawn_Object(obj)
        # Mutant = obj.MutateObject()

        voxels, voxel_corner = learner.object_to_voxels(obj)

        # get candidate grasp points
        cands = learner.collect_grasp_candidates(voxels)

        # convert back to simulator units
        coords = learner.voxel_to_sim_coords(cands, voxel_corner)

        # object may not be balanced on its own, run physics for a few seconds to let it settle in a stable pose
        orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
        exp.env.settle(exp.env.get_position(), seconds=3)
        rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

        # abort if object fell off of table
        if rest_pos[2] < table_height:
            pb.removeBody(obj_id)
            exp.env.close()
            continue
        # sys.exit(0)

        # transform grasp coordinates to object pose
        M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
        M = np.array(M).reshape(3, 3)
        rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
        interm_result = []
        #pb.removeBody(obj_id)
        Avg_result = 0
        for i in range(len(rest_coords)-1): # choosing all candidates
            if i > 0: # respawn object after an attempt.
                #exp.env.reset()
                exp.reset_robot()
                obj_id = exp.Spawn_Object(obj)
                orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
                exp.env.settle(exp.env.get_position(), seconds=2)
                rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

                # abort if object fell off of table
                if rest_pos[2] < table_height:
                    pb.removeBody(obj_id)
                    continue
                # sys.exit(0)

                # transform grasp coordinates to object pose
                M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
                M = np.array(M).reshape(3, 3)
                rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
            grip_points = rest_coords[i]
            trajectory = learner.get_pick_trajectory(exp.env, grip_points)

            # try best trajectory
            for angles in trajectory:
                exp.env.goto_position(angles, duration=2)
                exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
                # input('.')
            picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
            Num_Grips_attempted = Num_Grips_attempted + 1
            if picked_pos[2] > rest_pos[2]:
                print("\nSuccess , Total Attempt number ",Num_Grips_attempted )
                Num_success = Num_success + 1
            interm_result .append(picked_pos[2] - rest_pos[2])
            pb.removeBody(obj_id)
        Avg_result = np.sum(interm_result) / len(interm_result)
        obj_result.append(Avg_result)
        mutants = obj.Multiple_MutateObject()
        mutants = np.random.choice(mutants,15)
        Mutant_G1_result = []
        for mut in mutants:
            # obj = MultObjPick.Obj(mut.dims, n_parts, rgb)
            # obj.GenerateObject(dims, n_parts, [0, 0, 0])
            exp.reset_robot()
            obj_id = exp.Spawn_Object(mut)
            voxels, voxel_corner = learner.object_to_voxels(mut)
            # get candidate grasp points
            cands = learner.collect_grasp_candidates(voxels)
            # convert back to simulator units
            coords = learner.voxel_to_sim_coords(cands, voxel_corner)
            orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
            exp.env.settle(exp.env.get_position(),
                           seconds=3)  # wait for object to settle since its dropped from a small height
            rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)
            if rest_pos[2] < table_height:  # object falls/tumbles off , ignore this case
                pb.removeBody(obj_id)
                #exp.env.close()
                continue
                # transform grasp coordinates to object pose
                # transform grasp coordinates to object pose
            M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
            M = np.array(M).reshape(3, 3)
            rest_coords = np.dot(coords, M.T) + np.array(rest_pos)

            # # visualize grasp points in simulator
            # pb.addUserDebugPoints(rest_coords[0], [[0.,1.,0.]]*2, 25.0)

            # select highest grasp coordinates
            # (heuristic to avoid object-gripper collision in top-down grasps)
            hi = rest_coords.mean(axis=1)[:, 2].argmax()
            G1_interm_result = []
            for i in range(len(rest_coords)-1):  # choosing all candidates
                if i > 0:  # respawn object after an attempt.
                    exp.reset_robot()
                    obj_id = exp.Spawn_Object(mut)
                    orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
                    exp.env.settle(exp.env.get_position(), seconds=2)
                    rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

                    # abort if object fell off of table
                    if rest_pos[2] < table_height:
                        pb.removeBody(obj_id)
                        continue
                    # sys.exit(0)

                    # transform grasp coordinates to object pose
                    M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
                    M = np.array(M).reshape(3, 3)
                    rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
                grip_points = rest_coords[i]
                trajectory = learner.get_pick_trajectory(exp.env, grip_points)

                # try best trajectory
                for angles in trajectory:
                    exp.env.goto_position(angles, duration=2)
                    exp.env.goto_position(angles, duration=.1)  # in case it needs a little more time to converge
                    # input('.')
                picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
                Num_Grips_attempted = Num_Grips_attempted + 1
                if picked_pos[2] > rest_pos[2]:
                    print("\nSuccess , Total Attempt number ",Num_Grips_attempted )
                    Num_success = Num_success + 1
                G1_interm_result.append((picked_pos[2] - rest_pos[2]))
                pb.removeBody(obj_id)
            Avg_result = np.sum(G1_interm_result) / len(G1_interm_result)
            G1_interm_result.clear()
            Mutant_G1_result.append(Avg_result)
        with open(f'Gen1_results.pickle','ab+') as handle:
            pickle.dump(Mutant_G1_result,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        Top4Index = sorted(range(len(Mutant_G1_result)), key=lambda i: Mutant_G1_result[i], reverse=True)[-1:]
        # Best Mutant candidates for next generation / Least grippable objects
        Overall_Mutants_result = []
        num_generations =15
        Generation_results =[]
        for i in range(num_generations):
            Mutant_newgen_result = []
            #Top4Index_RC = np.random.choice(Top4Index,1)
            NewParents = [mutants[j] for j in Top4Index]
            mutants.tolist().clear()
            g_result = []
            g_result.clear()
            for parent in NewParents:
                p_mutants = parent.Multiple_MutateObject()
                #mutants = mutants+p_mutants
                mutant_avg =[]
                mutant_avg.clear()
                mutant_error = []
                p_mutants = np.random.choice(p_mutants,10)
                for child in p_mutants:
                    obj_id = exp.Spawn_Object(child)
                    voxels, voxel_corner = learner.object_to_voxels(child)
                    cands = learner.collect_grasp_candidates(voxels)
                    coords = learner.voxel_to_sim_coords(cands, voxel_corner)
                    orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
                    exp.env.settle(exp.env.get_position(),seconds=3)  # wait for object to settle since its dropped from a small height
                    rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)
                    if rest_pos[2] < table_height:  # object falls/tumbles off , ignore this case
                        pb.removeBody(obj_id)
                        #exp.env.close()
                        continue
                    M = pb.getMatrixFromQuaternion(rest_orn)  # orientation of rest object in world coordinates
                    M = np.array(M).reshape(3, 3)
                    rest_coords = np.dot(coords, M.T) + np.array(rest_pos)
                    OneMutant_result = []
                    for j in range(len(rest_coords)-1):

                        if j > 0:  # respawn object after an attempt.
                            exp.reset_robot()
                            obj_id = exp.Spawn_Object(child)
                            orig_pos, orig_orn = pb.getBasePositionAndOrientation(obj_id)
                            exp.env.settle(exp.env.get_position(), seconds=2)
                            rest_pos, rest_orn = pb.getBasePositionAndOrientation(obj_id)

                            # abort if object fell off of table
                            if rest_pos[2] < table_height:
                                pb.removeBody(obj_id)
                                continue
                        grip_points = rest_coords[j]
                        trajectory = learner.get_pick_trajectory(exp.env, grip_points)
                        for angles in trajectory:
                            exp.env.goto_position(angles, duration=2)
                            exp.env.goto_position(angles,duration=.1)  # in case it needs a little more time to converge

                        picked_pos, _ = pb.getBasePositionAndOrientation(obj_id)
                        mutant_error.append((picked_pos[2] - rest_pos[2]))
                        pb.removeBody(obj_id)
                    mutant_avg.append(np.min(mutant_error))
                g_result.append(mutant_avg)

            Generation_results.append(g_result)
            Top4Index = sorted(range(len(g_result)), key=lambda i: g_result[i], reverse=True)[-1:]
            with open(f'Generational_{i}_results.pickle','ab+') as handle:
                pickle.dump(Generation_results,handle,protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
        exp.env.close()
    with open('Obj.pickle', 'ab+') as handle:
        pickle.dump(obj_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


import pickle
from Meshtovoxel import Voxelize
if __name__ == "__main__":
    import multiprocessing as mp
    import MultObjPick
    #import pybullet as pb

    voxel_size = 0.015  # dimension of each voxel
    num_prll_prcs = 5
    dims = voxel_size * np.ones(3) / 2  # dims are actually half extents
    n_parts = 45
    rgb = [(.75, .25, .25)] * n_parts
    gen0_results = []
    obj = MultObjPick.Obj(dims, n_parts, rgb)
    obj.GenerateObject(dims, n_parts, [0, 0, 0])
    interm_pos = Voxelize("F6.obj")
    r = np.array(interm_pos)
    result = r*voxel_size
    obj.positions = result

    # obj_id = exp.Spawn_Object(obj)
    # Mutant = obj.MutateObject()
    Gen0_obj1_res , num_grips = AttemptGrips(obj,0)

    gen0_results.append(Gen0_obj1_res)
    while Gen0_obj1_res<=0:
        print("Redo obj1 , fell off the table or bad object")
        obj = MultObjPick.Obj(dims, n_parts, rgb)
        obj.GenerateObject(dims, n_parts, [0, 0, 0])
        obj.positions = result
        Gen0_obj1_res, num_grips2 = AttemptGrips(obj, 0)
    print("Completed gen 0 original Parent1")

    obj2 = MultObjPick.Obj(dims, n_parts, rgb)
    obj2.GenerateObject(dims, n_parts, [0, 0, 0])

    Gen0_obj2_res , num_grips2 = AttemptGrips(obj2,0)
    while Gen0_obj2_res<=0:
        print("Redo obj2 , fell off the table or bad object")
        obj2 = MultObjPick.Obj(dims, n_parts, rgb)
        obj2.GenerateObject(dims, n_parts, [0, 0, 0])
        Gen0_obj2_res, num_grips2 = AttemptGrips(obj2, 0)

    print("Completed gen 0 original Parent2")
    gen0_results.append(Gen0_obj2_res)
    numgrip = []
    numgrip.append(num_grips)
    numgrip.append(num_grips2)
    with open(f'mutantGen_0_results.pickle', 'ab+') as handle:
        pickle.dump(gen0_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(f'mutant_Gen0_numgrip_results.pickle', 'ab+') as handle:
        pickle.dump(numgrip, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print("\n Completed Gen0")
    num_gen = 30
    mutantres = []
    grip_per_gen = []
    for i in range(1,num_gen):
        grip_gen = 0
        print("\n Working on Gen",i)
        mutantres.clear()
        child = obj.crossover(obj,obj2,int(n_parts/2))
        mutants = child.Multiple_MutateObject()
        mutant_sample = np.random.choice(mutants,20)
        mutant_sample=np.append(mutant_sample,obj)
        mutant_sample=np.append(mutant_sample, obj2)
        print("\n Choose 20 random mutants - Completed")
        grips_res = []
        grips_res.clear()
        for child in mutant_sample:
            m_res,succ_grips = AttemptGrips(child,i)
            print("\nC")
            if m_res != -1:
                grip_per_gen.append(succ_grips)

                mutantres.append(m_res)
        if len(mutantres) == 0:
            temp = obj.copy()
            obj = obj2.copy()
            obj2 = temp.copy()
            continue
        with open(f'mutantGen_{i}_results.pickle', 'ab+') as handle:
            pickle.dump(mutantres, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        with open(f'mutantGen_{i}_numgrips_results.pickle', 'ab+') as handle:
            pickle.dump(grip_per_gen, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        grips_res.append(grip_per_gen)
        Parents_Index= sorted(range(len(mutantres)), key=lambda i: mutantres[i], reverse=True)[-2:]
        obj = mutant_sample[Parents_Index[0]]
        obj2 = mutant_sample[Parents_Index[1]]
        parentlist = list()
        parentlist.clear()
        parentlist.append(obj)
        parentlist.append(obj2)
        with open(f'mutantGenParent_{i}_results.pickle', 'ab+') as handle:
            pickle.dump(mutantres, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


        parentlist.clear()
    with open(f'mutantGen_numgrips_gen_results.pickle', 'ab+') as handle:
        pickle.dump(grips_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

print()

#pickle to load/save graphs