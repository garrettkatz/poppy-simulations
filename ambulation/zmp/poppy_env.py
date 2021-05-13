import pybullet as pb
import pybullet_data as pbd
import time
import numpy as np
import geom
import torch

class PoppyEnv(object):
    def __init__(self, g = 9.81, init_pos = (0, 0, 0.41), init_ori = (180, 0, 90), dT = 5e-3, poppy_path = 'E:/Syracuse/Robot/robot/code/poppy-standing-pybullet/poppyhumanoid/Poppy_Humanoid_fixed.URDF'):
        pb.connect(pb.GUI)

        gravity = (0, 0, -g)
        pb.setGravity(*gravity)

        pb.setAdditionalSearchPath(pbd.getDataPath())
        self.planeid = pb.loadURDF('plane.urdf')
        pb.setAdditionalSearchPath('./')
        self.poppyid = pb.loadURDF(poppy_path, basePosition = init_pos, baseOrientation = geom.euler2quaternion(*init_ori))

        self.dT = dT
        self.init_base_pos = np.array(init_pos)
        self.init_base_ori = geom.euler2quaternion(*init_ori)
        
        '''
            Poppy information
        '''
        # Revolute joints: [(0, b'r_hip_x'), (1, b'r_hip_z'), (2, b'r_hip_y'), (3, b'r_knee_y'), (4, b'r_ankle_y'), (5, b'l_hip_x'), (6, b'l_hip_z'), (7, b'l_hip_y'), (8, b'l_knee_y'), (9, b'l_ankle_y')]
        # hip_x, hip_z, hip_y, knee_y, ankle_y
        self.right_joints = [0, 1, 2, 3, 4]
        self.left_joints = [5, 6, 7, 8, 9]

        self.right_joint_axis = np.array([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 0, 0]
        ])

        self.left_joint_axis = np.array([
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 0],
            [-1, 0, 0], 
            [-1, 0, 0]
        ])

        self.right_joint_xyz = np.array([
            [-0.0225417390633467, 0, 0],
            [-0.0439986111539757, 0, 0.005],
            [0, -0.024, 0],
            [0, -0.182, 0],
            [0, -0.18, 0]
        ])

        self.left_joint_xyz = np.array([
            [0.0225417390633466, 0, 0],
            [0.0439986111539757, 0, 0.005],
            [0, -0.024, 0],
            [0, -0.182, 0],
            [0, -0.18, 0]
        ])

        self.right_joint_rpy = np.array([
            [1.5707963267949, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1.5707963267949, 0],
            [0, -1.5708, 0]
        ])

        self.left_joint_rpy = np.array([
            [1.5707963267949, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1.5707963267949, 0],
            [0, -1.5708, 0]
        ])

        self.right_joint_axis = self.right_joint_axis[:, [2, 0, 1]]
        self.left_joint_axis = self.left_joint_axis[:, [2, 0, 1]]
        self.right_joint_xyz = self.right_joint_xyz[:, [2, 0, 1]]
        self.left_joint_xyz = self.left_joint_xyz[:, [2, 0, 1]]
        self.right_joint_rpy = self.right_joint_rpy[:, [2, 0, 1]]
        self.left_joint_rpy = self.left_joint_rpy[:, [2, 0, 1]]
        
        # print(geom.GetRotFromAngles(self.right_joint_rpy[0]) @ self.right_joint_axis[0])
        # print(geom.GetRotFromAngles(self.right_joint_rpy[1]) @ self.right_joint_axis[1])
        # print(geom.GetRotFromAngles(self.right_joint_rpy[2]) @ self.right_joint_axis[2])
        # print(geom.GetRotFromAngles(self.right_joint_rpy[3]) @ self.right_joint_axis[3])
        # print(geom.GetRotFromAngles(self.right_joint_rpy[4]) @ self.right_joint_axis[4])
        # print()
        # print(geom.GetRotFromAngles(self.left_joint_rpy[0]) @ self.left_joint_axis[0])
        # print(geom.GetRotFromAngles(self.left_joint_rpy[1]) @ self.left_joint_axis[1])
        # print(geom.GetRotFromAngles(self.left_joint_rpy[2]) @ self.left_joint_axis[2])
        # print(geom.GetRotFromAngles(self.left_joint_rpy[3]) @ self.left_joint_axis[3])
        # print(geom.GetRotFromAngles(self.left_joint_rpy[4]) @ self.left_joint_axis[4])

        # self.right_joint_xyz = np.array([pb.getJointInfo(self.poppyid, i)[-3] for i in self.right_joints])
        # self.left_joint_xyz = np.array([pb.getJointInfo(self.poppyid, i)[-3] for i in self.left_joints])
        # self.right_joint_xyz = self.right_joint_xyz[:, [2, 0, 1]]
        # self.left_joint_xyz = self.left_joint_xyz[:, [2, 0, 1]]
        # print(self.right_joint_xyz)
        # print(self.left_joint_xyz)

        # self.right_joint_axis = np.array([pb.getJointInfo(self.poppyid, i)[-4] for i in self.right_joints])
        # self.left_joint_axis = np.array([pb.getJointInfo(self.poppyid, i)[-4] for i in self.left_joints])
        # print(self.right_joint_axis)
        # print(self.left_joint_axis)

        self._lambda = 1

        # The length of thighs and shanks
        self.thigh_len, self.shank_len = 0.182, 0.18
        self.CoM_height = 0.41
        self.foot_to_CoM_dy = 0.04
        self.left_D = self.right_D = 0.06654

        self.max_force_for_joints = [10] * len(self.left_joints)

        self.record = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, 'test.mp4')

    def __del__(self):
        pb.stopStateLogging(self.record)

    
    def setLeftJointPositions(self, target_pos):
        pb.setJointMotorControlArray(self.poppyid, self.left_joints, pb.POSITION_CONTROL, targetPositions = target_pos)

    def getLeftJointPositions(self):
        return [pb.getJointState(self.poppyid, joint)[0] for joint in self.left_joints]

    def setRightJointPositions(self, target_pos):
        pb.setJointMotorControlArray(self.poppyid, self.right_joints, pb.POSITION_CONTROL, targetPositions = target_pos)

    def getRightJointPositions(self):
        return [pb.getJointState(self.poppyid, joint)[0] for joint in self.right_joints]

    def setAllJointPositions(self, target_pos):
        pb.setJointMotorControlArray(self.poppyid, self.right_joints + self.left_joints, pb.POSITION_CONTROL, targetPositions = target_pos)

    # def initialize(self, left_joint_pos, right_joint_pos, init_time = 0.5, idle_time = 1):
    #     wait_steps = int(init_time / self.dT)
    #     time_steps = np.linspace(0, 1, num = wait_steps, dtype = np.float32)
    #     for t in time_steps:
    #         self.setLeftJointPositions(left_joint_pos * t)
    #         self.setRightJointPositions(right_joint_pos * t)
    #         self.step()

    #     for _ in range(int(idle_time / self.dT)):
    #         self.setLeftJointPositions(left_joint_pos)
    #         self.setRightJointPositions(right_joint_pos)
    #         self.step()

    def initialize(self, left_joint_pos, right_joint_pos, init_time = 1):
        init_steps = np.arange(0, int(init_time / self.dT))

        for i in init_steps:
            self.setLeftJointPositions(left_joint_pos)
            self.setRightJointPositions(right_joint_pos)
            pb.resetBasePositionAndOrientation(self.poppyid, self.init_base_pos, self.init_base_ori)
            self.step()

    def step(self):
        # poppy_pos, _ = pb.getBasePositionAndOrientation(self.poppyid)
        # pb.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw = 135, cameraPitch = -10, cameraTargetPosition = poppy_pos)
        pb.stepSimulation()
        time.sleep(self.dT)

    '''
        Geometric
    '''
    # def inverse_kinematics(self, foot_pos, com_pos, is_left = False):
    #     ''' 
    #         Inverse kinematics function.
    #         l3: thigh length
    #         l4: shank length
    #     '''
    #     CoMx, CoMy, CoMz = com_pos
    #     fx, fy, fz = foot_pos
    #     dx, dy, dz = fx - CoMx, fy - CoMy, CoMz - fz

    #     a = np.sqrt(dx**2 + dy**2 + dz**2)
    #     theta2 = np.pi - np.arccos((self.thigh_len**2 + self.shank_len**2 - a**2) / (2 * self.thigh_len * self.shank_len))
    #     theta5 = np.arccos(dz / a)
    #     theta1 = np.arcsin(self.shank_len * np.sin(theta2) / a) + theta5
    #     theta3 = theta2 - theta1
    #     theta4 = np.arctan(dy / dz)

    #     # hip_x, hip_z, hip_y, knee_y, ankle_y
    #     angles = np.array([theta4, 0, theta1, theta2, theta3])

    #     # angles = np.array([0, theta4, theta1, theta2, theta3, 0])

    #     # Direction
    #     if is_left:
    #         angles *= np.array([-1, 1, 1, -1, 1])
    #     else:
    #         angles *= np.array([-1, 1, 1, -1, 1])

    #     return angles

    '''
        Jacobian
    '''
    def get_leg_transmat(self, joint_pos, is_left = False):
        joint_axis = self.left_joint_axis if is_left else self.right_joint_axis
        joint_xyz = self.left_joint_xyz if is_left else self.right_joint_xyz
        joint_rpy = self.left_joint_rpy if is_left else self.right_joint_rpy

        hip_x_axis, hip_z_axis, hip_y_axis, knee_y_axis, ankle_y_axis = joint_axis
        hip_x_xyz, hip_z_xyz, hip_y_xyz, knee_y_xyz, ankle_y_xyz = joint_xyz
        hip_x_rpy, hip_z_rpy, hip_y_rpy, knee_y_rpy, ankle_y_rpy = joint_rpy
        hip_x_pos, hip_z_pos, hip_y_pos, knee_y_pos, ankle_y_pos = joint_pos

        # T_hip_x = geom.RotToTrans(geom.RodriguesFormula(geom.GetRotFromAngles(hip_x_rpy) @ hip_x_axis, hip_x_pos), hip_x_xyz)
        # T_hip_z = T_hip_x @ geom.RotToTrans(geom.RodriguesFormula(geom.GetRotFromAngles(hip_z_rpy) @ hip_z_axis, hip_z_pos), hip_z_xyz)
        # T_hip_y = T_hip_z @ geom.RotToTrans(geom.RodriguesFormula(geom.GetRotFromAngles(hip_y_rpy) @ hip_y_axis, hip_y_pos), hip_y_xyz)
        # T_knee_y = T_hip_y @ geom.RotToTrans(geom.RodriguesFormula(geom.GetRotFromAngles(knee_y_rpy) @ knee_y_axis, knee_y_pos), knee_y_xyz)
        # T_ankle_y = T_knee_y @ geom.RotToTrans(geom.RodriguesFormula(geom.GetRotFromAngles(ankle_y_rpy) @ ankle_y_axis, ankle_y_pos), ankle_y_xyz)

        T_hip_x = geom.RotToTrans(geom.RodriguesFormula(hip_x_axis, hip_x_pos), hip_x_xyz)
        T_hip_z = T_hip_x @ geom.RotToTrans(geom.RodriguesFormula(hip_z_axis, hip_z_pos), hip_z_xyz)
        T_hip_y = T_hip_z @ geom.RotToTrans(geom.RodriguesFormula(hip_y_axis, hip_y_pos), hip_y_xyz)
        T_knee_y = T_hip_y @ geom.RotToTrans(geom.RodriguesFormula(knee_y_axis, knee_y_pos), knee_y_xyz)
        T_ankle_y = T_knee_y @ geom.RotToTrans(geom.RodriguesFormula(ankle_y_axis, ankle_y_pos), ankle_y_xyz)

        return T_hip_x, T_hip_z, T_hip_y, T_knee_y, T_ankle_y

    def forward_kinematics(self, joint_pos, is_left = False):
        T_end_effector = self.get_leg_transmat(joint_pos, is_left)[-1]
        return geom.GetRotAndPosFromTrans(T_end_effector)

    def jacobian(self, q, is_left = False):
        Ts = self.get_leg_transmat(q, is_left)
        joint_axis = self.left_joint_axis if is_left else self.right_joint_axis
        Rs, ps = [], []

        for T in Ts:
            R, p = geom.GetRotAndPosFromTrans(T)
            Rs.append(R)
            ps.append(p)

        Ws = [R @ axis for R, axis in zip(Rs, joint_axis)]

        J = np.vstack((np.hstack((np.cross(w, ps[-1] - p), w)) for w, p in zip(Ws, ps))).T
        
        return J
    
    def inverse_kinematics(self, zmp_ref, omega_ref = np.array([0, 0, 0]), is_left = False):
        q = self.getLeftJointPositions() if is_left else self.getRightJointPositions()
        R, p = self.forward_kinematics(q, is_left)
        omega = np.array(geom.GetRollPitchYawFromRot(R))

        # p: joint position  omega: joint velocity
        dp, domega = zmp_ref - p, omega_ref - omega
        dp_domega = np.append(dp, domega)

        J = self.jacobian(q, is_left)
        dq = self._lambda * np.linalg.pinv(J).dot(dp_domega)

        return q + dq

if __name__ == '__main__':
    env = PoppyEnv()
    for _ in range(200):
        env.step()