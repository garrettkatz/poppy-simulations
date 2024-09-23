import pybullet as pb
from pybullet_data import getDataPath
import time
import numpy as np

def nop_step_hook(env, action): return None

class PoppyEnv(object):

    # override this for urdf logic, should return robot pybullet id
    def load_urdf(self, use_fixed_base=False):
        return 0

    def __init__(self,
        control_mode=pb.POSITION_CONTROL,
        timestep=1/240,
        control_period=1,
        show=True,
        step_hook=None,
        use_fixed_base=False,
        use_self_collision=False,
    ):

        # step_hook(env, action) is called in each env.step(action)
        if step_hook is None: step_hook = nop_step_hook

        self.control_mode = control_mode
        self.timestep = timestep
        self.control_period = control_period
        self.show = show
        self.step_hook = step_hook

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        if show: pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        self.ground_id = pb.loadURDF("plane.urdf")
        
        # use overridden loading logic
        self.robot_id = self.load_urdf(use_fixed_base, use_self_collision)

        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_name, self.joint_index, self.joint_fixed = {}, {}, {}
        self.joint_low = np.empty(self.num_joints)
        self.joint_high = np.empty(self.num_joints)
        self.joint_parent = np.empty(self.num_joints, dtype=int)
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
            self.joint_fixed[i] = (info[2] == pb.JOINT_FIXED)
            self.joint_low[i] = info[8]
            self.joint_high[i] = info[9]
            self.joint_parent[i] = info[-1]
        
        self.initial_state_id = pb.saveState(self.client_id)
    
    def reset(self):
        # pb.restoreState(stateId = self.initial_state_id, clientServerId = self.client_id) # quickstart has wrong keyword
        pb.restoreState(stateId = self.initial_state_id, physicsClientId = self.client_id) # pybullet.c shows this one
    
    def close(self):
        pb.disconnect()
        
    def step(self, action=None, sleep=None):
        
        self.step_hook(self, action)

        if action is not None:
            duration = self.control_period * self.timestep
            distance = np.fabs(action - self.get_position())
            pb.setJointMotorControlArray(
                self.robot_id,
                jointIndices = range(len(self.joint_index)),
                controlMode = self.control_mode,
                targetPositions = action,
                targetVelocities = [0]*len(action),
                positionGains = [.25]*len(action), # important for constant position accuracy
                #maxVelocities = distance / duration,
            )

        if sleep is None: sleep = self.show
        if sleep:
            for _ in range(self.control_period):
                start = time.perf_counter()
                pb.stepSimulation()
                duration = time.perf_counter() - start
                remainder = self.timestep - duration
                if remainder > 0: time.sleep(remainder)
        else:
            for _ in range(self.control_period):
                pb.stepSimulation()

    # base position/orientation and velocity/angular
    def get_base(self):
        pos, orn = pb.getBasePositionAndOrientation(self.robot_id)
        vel, ang = pb.getBaseVelocity(self.robot_id)
        return pos, orn, vel, ang
    def set_base(self, pos=None, orn=None, vel=None, ang=None):
        _pos, _orn, _vel, _ang = self.get_base()
        if pos == None: pos = _pos
        if orn == None: orn = _orn
        if vel == None: vel = _vel
        if ang == None: ang = _ang
        pb.resetBasePositionAndOrientation(self.robot_id, pos, orn)
        pb.resetBaseVelocity(self.robot_id, vel, ang)
    
    # get/set joint angles as np.array
    def get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])    
    def set_position(self, position):
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def angle_array(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array
    # convert back from dict to array
    def angle_dict(self, angle_array, convert=True):
        return {
            name: angle_array[j] * 180/np.pi
            for j, name in enumerate(self.joint_index)}

    # convert trajectory of radian angle arrays to pypot degree angle dictionaries
    #     trajectory[t]: (duration, target angle array)
    def get_pypot_trajectory(self, trajectory):
    
        pypot_trajectory = []
        for (duration, angles) in trajectory:
            angle_dict = self.angle_dict(angles)
            pypot_trajectory.append((duration, angle_dict))
        return tuple(pypot_trajectory)

    # pypot-style command, goes to target joint position with given speed
    # target is a joint angle array
    # speed is desired joint speed
    # duration takes precedence over speed if not None
    # if hang==True, wait for user enter at each timestep of motion
    def goto_position(self, target, speed=1., duration=None, hang=False):

        current = self.get_position()
        distance = np.sum((target - current)**2)**.5
        if duration == None: duration = distance / speed

        num_steps = int(duration / (self.timestep * self.control_period) + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        positions = np.empty((num_steps, self.num_joints))
        for a, action in enumerate(trajectory):
            self.step(action)
            positions[a] = self.get_position()
            if hang: input('..')

        return positions

    # Return current Cartesian locations of joints
    def forward_kinematics(self, angles=None):
        # update to new angles if provided
        if angles is not None: self.set_position(angles)

        jnt_loc = np.zeros((self.num_joints, 3))
        for idx in range(self.num_joints):
            link_state = pb.getLinkState(self.robot_id, idx)
            jnt_loc[idx] = link_state[4]

        return jnt_loc

    # Return current center of mass
    def center_of_mass(self, angles=None):
        # update to new angles if provided
        if angles is not None: self.set_position(angles)

        # initialize from base
        total_mass = pb.getDynamicsInfo(self.robot_id, -1)[0]
        com_loc = np.array(self.get_base()[0]) * total_mass

        # accumulate over links
        for idx in range(self.num_joints):
            link_com = pb.getLinkState(self.robot_id, idx)[0]
            link_mass = pb.getDynamicsInfo(self.robot_id, idx)[0]

            com_loc += np.array(link_com) * link_mass
            total_mass += link_mass

        return com_loc / total_mass

    # Run IK, accounting for fixed joints
    def inverse_kinematics(self, link_indices, target_positions, num_iters=1000, resid_thresh=1e-4):
        # targets for link coordinates, not COM coordinates

        angles = pb.calculateInverseKinematics2(
            self.robot_id,
            link_indices,
            target_positions,
            # lowerLimits = [self.joint_low[j] for j in range(self.num_joints) if not self.joint_fixed[j]],
            # upperLimits = [self.joint_high[j] for j in range(self.num_joints) if not self.joint_fixed[j]],
            residualThreshold=resid_thresh, # default 1e-4 not enough for ergo jr
            maxNumIterations=num_iters, # default 20 usually not enough
        )

        a = 0
        result = self.get_position()
        for r in range(self.num_joints):
            if not self.joint_fixed[r]:
                result[r] = angles[a]
                a += 1
        
        return result

    # Invert part of the kinematic chain while the rest is fixed
    def partial_ik(self, links, targets, angles, free, num_iters=1000, resid_thresh=1e-4, verbose=False):
        # links: list of joint indices, passed to self.inverse_kinematics
        #   these joint (not CoM) locations must reach the targets
        # targets: array of targets, passed to self.inverse_kinematics
        #   these are the target locations for the links given above
        # angles: a joint angle array
        #   most of these angles must remain fixed in the solution
        # free: a list of joint indices
        #   locations of these joints are unconstrained
        #   their parent joints are the only ones whose angles can be changed in the solution
        # returns full solution joint angle array, out-of-joint-limit flag, and maximum absolute target error

        # make sure links with targets are not also listed as free
        assert not any([j in links for j in free])

        # temporarily overwrite current joint angles
        save_angles = self.get_position()
        self.set_position(angles)

        # keep all non-free joints' parents fixed by using their current locations as additional targets
        all_targets = self.forward_kinematics()
        all_targets[links] = targets
        all_links = np.array([j for j in range(self.num_joints) if j not in free])
        all_targets = all_targets[all_links]

        # run IK with all constraints
        angles = self.inverse_kinematics(all_links, all_targets, num_iters=num_iters)

        # guard against occasional out-of-joint-limit solutions
        # assert(all([
        #     (self.joint_low[i] <= angles[i] <= self.joint_high[i]) or self.joint_fixed[i]
        #     for i in range(self.num_joints)]))
        oojl = False
        for i in range(self.num_joints):
            if not ((self.joint_low[i] <= angles[i] <= self.joint_high[i]) or self.joint_fixed[i]):
                if verbose:
                    print(f"{self.joint_name[i]}: {angles[i]} not in [{self.joint_low[i]}, {self.joint_high[i]}]!")
                oojl = True
        if oojl and verbose: input('uh oh...')

        # measure constraint error
        self.set_position(angles)
        all_actual = self.forward_kinematics()
        all_actual = all_actual[all_links]
        error = np.fabs(all_targets - all_actual).max()

        if verbose:
            print('iksolve errors:')
            print([self.joint_name[i] for i in range(self.num_joints) if i not in free])
            print(all_targets - all_actual)
            print(error)

        # restore original angles
        self.set_position(save_angles)

        # return result
        return angles, oojl, error

    # assign new pose and run simulation steps to gauge stability
    def settle(self, angles, base=None, seconds=1):
        # base: if not provided, use current base
        # seconds: simulation time, set to 0 to skip simulation
        # returns locations of link CoMs, joints, and base

        # set pose
        self.set_position(angles)
        if base is not None: self.set_base(*base)

        # simulate to equilibrium
        # step duration = control_period * timestep
        # num steps = duration / step duration
        num_steps = seconds / (self.control_period * self.timestep)
        for _ in range(int(num_steps)): self.step(angles)

        # return pose at equilibrium
        CoM = self.center_of_mass()
        jnt_loc = self.forward_kinematics()
        base = self.get_base()
        return CoM, jnt_loc, base

