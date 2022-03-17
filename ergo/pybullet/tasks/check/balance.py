import pybullet as pb
import pybullet_data as pbd
import time
import numpy as np
from collections import defaultdict

np.set_printoptions(suppress = True)

def GetJointObservation(robot_id, joints):
    return pb.getJointStates(robot_id, joints)


def compute_jacobian(robot_id, link_id, joint_states):
    all_joint_angles = [state[0] for state in joint_states]
    zero_vec = [0] * len(all_joint_angles)
    jv, _ = pb.calculateJacobian(robot_id, link_id, (0, 0, 0), all_joint_angles, zero_vec, zero_vec)
    jacobian = np.array(jv)
    assert jacobian.shape[0] == 3
    return jacobian


def get_sensor_data(robot_id, joint_indices):
    sensor_data = {}

    base_pos, base_ori = pb.getBasePositionAndOrientation(robot_id)
    sensor_data['base_pos'] = list(base_pos)
    sensor_data['base_ori'] = list(base_ori)
    sensor_data['base_rotmat'] = np.array(pb.getMatrixFromQuaternion(base_ori)).reshape(3, 3)

    base_linvel, base_angvel = pb.getBaseVelocity(robot_id)
    sensor_data['base_linvel'] = list(base_linvel)
    sensor_data['base_angvel'] = list(base_angvel)

    joint_info = list(zip(*pb.getJointStates(robot_id, joint_indices)))
    sensor_data['joint_pos'] = np.array(joint_info[0])
    sensor_data['joint_vel'] = np.array(joint_info[1])

    return sensor_data


def get_jacobian(robot_id, link_id, sensor_data):
    local_inertial_frame_pos = pb.getLinkState(robot_id, link_id)[2]
    local_inertial_frame_pos = [0, 0, 0]
    jpos = sensor_data['joint_pos'].tolist()
    jvel = [0.0] * len(jpos)
    # jvel = sensor_data['joint_vel'].tolist()
    jacc = [0.0] * len(jpos)
    J_lin, J_ang = pb.calculateJacobian(robot_id, link_id, local_inertial_frame_pos, jpos, jvel, jacc)
    return np.array(J_lin), np.array(J_ang)


def get_mass_matrix(robot_id, sensor_data):
    jpos = sensor_data['joint_pos'].tolist()
    mass_mat = pb.calculateMassMatrix(robot_id, jpos)
    return np.array(mass_mat)


def get_inverse_dynamics_matrix(robot_id, sensor_data):
    jpos = sensor_data['joint_pos'].tolist()
    jvel = sensor_data['joint_vel'].tolist()
    jacc = [0.0] * len(jpos)
    inverse_dynamics_mat = pb.calculateInverseDynamics(robot_id, jpos, jvel, jacc)
    return np.array(inverse_dynamics_mat)


def get_spatial_velocity(robot_id, link_id):
    link_state = pb.getLinkState(robot_id, link_id, computeLinkVelocity = True, computeForwardKinematics = True)
    linvel = np.array(link_state[6])
    angvel = np.array(link_state[7])
    return linvel, angvel


def get_ground_contact_forces(robot_id, plane_id):
    reaction_forces = defaultdict(lambda: [])

    contact_points_info = pb.getContactPoints(robot_id, plane_id)
    for info in contact_points_info:
        contact_link_idx = info[3]
        contact_position_on_link = info[5]
        contact_normal = info[7]
        contact_force = info[9]
        reaction_forces[contact_link_idx].append( (np.array(contact_position_on_link), contact_force * np.array(contact_normal)) )
    
    return reaction_forces


def map_contact_force_to_torque(robot_id, reaction_forces, sensor_data):
    tau = np.zeros(37)
    taus = []

    for link_id, force_list in reaction_forces.items():
        J_lin, _ = get_jacobian(robot_id, link_id, sensor_data)
        for _, force in force_list:
            tau += J_lin.T @ force
            taus.append(J_lin.T @ force)

    return tau


def get_com(robot_id):
    num_joints = pb.getNumJoints(robot_id)
    masses = []
    for i in range(-1, num_joints):
        masses.append(pb.getDynamicsInfo(robot_id, i)[0])
    masses = np.expand_dims(np.array(masses), 1)

    link_com_pos = []
    base_pos, _ = pb.getBasePositionAndOrientation(robot_id)
    link_com_pos.append(base_pos)
    for i in range(num_joints):
        link_com_pos.append(pb.getLinkState(robot_id, i)[0])
    link_com_pos = np.array(link_com_pos)

    com = np.sum(masses * link_com_pos, 0) / np.sum(masses)
    return com


def compute_zmp(reaction):
    if len(reaction.keys()) == 0:
        return np.zeros(3)

    numerator_x, numerator_y = 0, 0
    denominator = 0

    for _, force_list in reaction.items():
        for pos, force in force_list:
            numerator_x += pos[0] * force[2]
            numerator_y += pos[1] * force[2]
            denominator += force[2]

    if denominator == 0:
        denominator = 1.0
    return np.array([numerator_x, numerator_y, 0]) / denominator


def visualization(com_visual, com_pos, zmp_visual, zmp_pos):
    pb.resetBasePositionAndOrientation(com_visual, com_pos, [0, 0, 0, 1])
    pb.resetBasePositionAndOrientation(zmp_visual, zmp_pos, [0, 0, 0, 1])



def init(robot_id):
    joints = []
    for i in range(pb.getNumJoints(robot_id)):
        joint_info = pb.getJointInfo(robot_id, i)
        if joint_info[2] != pb.JOINT_FIXED:
            joints.append((i, joint_info[1], joint_info[2]))
    # print(joints)
    joint_indices, joint_names, _ = list(zip(*joints))
    upperbody_joints = list(range(19, pb.getNumJoints(robot_id)))
    upperbody_original_joints = list(range(13, pb.getNumJoints(robot_id) - 6))

    for i in range(pb.getNumJoints(robot_id)):
        pb.changeDynamics(robot_id, i, linearDamping = 0, angularDamping = 0)

    # !!!!!
    #disable the default velocity motors
    #and set some position control with small force to emulate joint friction/return to a rest pose
    jointFrictionForce = 1
    for joint in range(pb.getNumJoints(robot_id)):
        pb.setJointMotorControl2(robot_id, joint, pb.POSITION_CONTROL, force=jointFrictionForce)
    # pb.setRealTimeSimulation(1)

    return joint_indices



def balance(robot_id, joint_indices, plane_id):
    sensor_data = get_sensor_data(robot_id, joint_indices)
    reaction_forces = get_ground_contact_forces(robot_id, plane_id)
    tau = map_contact_force_to_torque(robot_id, reaction_forces, sensor_data)
    N = get_inverse_dynamics_matrix(robot_id, sensor_data)

    com = get_com(robot_id)
    zmp = compute_zmp(reaction_forces)
    # visualization(com_visual, com, zmp_visual, zmp)

    pb.setJointMotorControlArray(robot_id, list(range(6, 10)), pb.TORQUE_CONTROL, forces = (N[6:10]).tolist())
    pb.setJointMotorControlArray(robot_id, list(range(11, 15)), pb.TORQUE_CONTROL, forces = (N[11:15]).tolist())

    pb.setJointMotorControl2(robot_id, 10, pb.TORQUE_CONTROL, force = N[10] + -20 * (curr_com_pos[1] - 0)) #  -5 * jpos[10] 
    pb.setJointMotorControl2(robot_id, 15, pb.TORQUE_CONTROL, force = N[15] + -20 * (curr_com_pos[1] - 0)) #  + -5 * jpos[15]
    pb.setJointMotorControl2(robot_id, 16, pb.TORQUE_CONTROL, force = N[16] + -10 * (curr_com_pos[1] - 0))
    pb.setJointMotorControl2(robot_id, 17, pb.TORQUE_CONTROL, force = N[17] + -10 * (curr_com_pos[0] - 0))

    pb.stepSimulation()




if __name__ == '__main__':
    poppy_path = '../../urdfs/ergo/fixed_base_poppy_ergo.pybullet.urdf'

    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    pb.setGravity(0, 0, -9.81)

    plane_id = pb.loadURDF("plane.urdf")
    robot_id = pb.loadURDF(poppy_path, (0, 0, 0.42), useFixedBase = True)
    com_sphere_id = pb.createVisualShape(pb.GEOM_SPHERE, radius = 0.02, rgbaColor = [1, 0, 0, 1])
    zmp_sphere_id = pb.createVisualShape(pb.GEOM_SPHERE, radius = 0.02, rgbaColor = [0, 0, 1, 1])
    com_visual = pb.createMultiBody(baseMass = 0, baseVisualShapeIndex = com_sphere_id, basePosition = [0, 0, 0])
    zmp_visual = pb.createMultiBody(baseMass = 0, baseVisualShapeIndex = zmp_sphere_id, basePosition = [0, 0, 0])

    

    '''
        [(0, b'joint0_trax', 1), (1, b'joint0_tray', 1), (2, b'joint0_traz', 1), 
        (3, b'joint0_rotx', 0), (4, b'joint0_roty', 0), (5, b'joint0_rotz', 0), 
        (6, b'r_hip_x', 0), (7, b'r_hip_z', 0), (8, b'r_hip_y', 0), 
        (9, b'r_knee_y', 0), (10, b'r_ankle_y', 0), (11, b'l_hip_x', 0), 
        (12, b'l_hip_z', 0), (13, b'l_hip_y', 0), (14, b'l_knee_y', 0), 
        (15, b'l_ankle_y', 0), (16, b'abs_y', 0), (17, b'abs_x', 0), 
        (18, b'abs_z', 0), (19, b'bust_y', 0), (20, b'bust_x', 0), 
        (21, b'head_z', 0), (22, b'head_y', 0), (24, b'l_shoulder_y', 0), 
        (25, b'l_shoulder_x', 0), (26, b'l_arm_z', 0), (27, b'l_elbow_y', 0), 
        (28, b'l_wrist_y', 0), (29, b'l_wrist_x', 0), (30, b'l_gripper', 0), 
        (33, b'r_shoulder_y', 0), (34, b'r_shoulder_x', 0), (35, b'r_arm_z', 0), 
        (36, b'r_elbow_y', 0), (37, b'r_wrist_y', 0), (38, b'r_wrist_x', 0), 
        (39, b'r_gripper', 0)]
    '''
    joints = []
    for i in range(pb.getNumJoints(robot_id)):
        joint_info = pb.getJointInfo(robot_id, i)
        if joint_info[2] != pb.JOINT_FIXED:
            joints.append((i, joint_info[1], joint_info[2]))
    # print(joints)
    joint_indices, joint_names, _ = list(zip(*joints))
    upperbody_joints = list(range(19, pb.getNumJoints(robot_id)))
    upperbody_original_joints = list(range(13, pb.getNumJoints(robot_id) - 6))

    for i in range(pb.getNumJoints(robot_id)):
        pb.changeDynamics(robot_id, i, linearDamping = 0, angularDamping = 0)

    # !!!!!
    #disable the default velocity motors
    #and set some position control with small force to emulate joint friction/return to a rest pose
    jointFrictionForce = 1
    for joint in range(pb.getNumJoints(robot_id)):
        pb.setJointMotorControl2(robot_id, joint, pb.POSITION_CONTROL, force=jointFrictionForce)
    # pb.setRealTimeSimulation(1)

    curr_com_pos = get_com(robot_id)
    com_vel = np.zeros(3)

    link_id = 15
    jpos_list = np.load('test.npy')


    for jpos in jpos_list:
        sensor_data = get_sensor_data(robot_id, joint_indices)
        reaction_forces = get_ground_contact_forces(robot_id, plane_id)

        # # Real velocity
        # real_linvel, real_angvel = get_spatial_velocity(robot_id, link_id)

        # # Compute the velocity by the jacobian matrix
        # J_lin, J_ang = get_jacobian(robot_id, link_id, sensor_data)
        # qdot = sensor_data['joint_vel']
        # # qdot[:3] = sensor_data['base_rotmat'].T @ qdot[:3]
        # # qdot[3:6] = sensor_data['base_rotmat'].T @ qdot[3:6]
        # # pred_linvel = sensor_data['base_rotmat'] @ J_lin @ qdot
        # # pred_angvel = sensor_data['base_rotmat'] @ J_ang @ qdot
        # pred_linvel = sensor_data['base_rotmat'] @ J_lin @ qdot
        # pred_angvel = sensor_data['base_rotmat'] @ J_ang @ qdot


        # M = get_mass_matrix(robot_id, sensor_data)
        N = get_inverse_dynamics_matrix(robot_id, sensor_data)


        # print('Linvel error: %.6f | Angvel error: %.6f' % (np.linalg.norm(real_linvel - pred_linvel), np.linalg.norm(real_angvel - pred_angvel)))
        # print(np.linalg.norm(real_linvel - pred_linvel), np.linalg.norm(real_angvel - pred_angvel), sensor_data['base_pos'])

        

        # M = get_mass_matrix(robot_id, sensor_data)
        # N = get_inverse_dynamics_matrix(robot_id, sensor_data)
        # tau = map_contact_force_to_torque(robot_id, reaction_forces, sensor_data)
        # qddot = np.linalg.inv(M) @ (tau - N)
        # # print(qddot)


        # q = sensor_data['joint_pos']
        # qdot = sensor_data['joint_vel']

        pb.setJointMotorControlArray(robot_id, upperbody_joints, pb.POSITION_CONTROL, jpos[upperbody_original_joints])

        # q = np.append(curr_com_pos[:-1], com_vel[:-1])
        # K_LQR = np.array([1, 1, 0.01, 0.01])
        # force = K_LQR @ q
        # print(q, force)
        # pb.setJointMotorControlArray(robot_id, [4, 9], pb.TORQUE_CONTROL, forces = [force, force])
        # pb.setJointMotorControlArray(robot_id, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12], pb.POSITION_CONTROL, [0.0] * 11)
        # pb.setJointMotorControlArray(robot_id, [0, 1, 2, 3, 5, 6, 7, 8, 12], pb.POSITION_CONTROL, [0.0] * 9)
        # q_r_ankle = sensor_data['joint_pos'][4] + 0.1 * curr_com_pos[1]
        # q_l_ankle = sensor_data['joint_pos'][9] + 0.1 * curr_com_pos[0]
        # pb.setJointMotorControlArray(robot_id, [4, 9], pb.POSITION_CONTROL, [q_r_ankle, q_l_ankle])
        # q_abs_y = sensor_data['joint_pos'][10] + 0.1 * curr_com_pos[1]
        # q_abs_x = sensor_data['joint_pos'][11] + 0.1 * curr_com_pos[0]
        # # pb.setJointMotorControlArray(robot_id, [10, 11], pb.POSITION_CONTROL, [q_abs_y, q_abs_x])



        pb.setJointMotorControlArray(robot_id, list(range(6, 19)), pb.TORQUE_CONTROL, forces = N[6:19].tolist())


        # pb.setJointMotorControlArray(robot_id, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15], pb.POSITION_CONTROL, [0.0] * 10)
        # pb.setJointMotorControlArray(robot_id, joint_indices[6:], pb.TORQUE_CONTROL, forces = 0)
        pb.stepSimulation()

        next_com_pos = get_com(robot_id)
        com_vel = (next_com_pos - curr_com_pos) / (1 / 240.0)
        curr_com_pos = next_com_pos


        time.sleep(0.03) 


    joint_debug_params = {}
    for i in joint_indices[19:]:
        joint_info = pb.getJointInfo(robot_id, i)
        joint_state = pb.getJointState(robot_id, i)
        joint_debug_params[i] = pb.addUserDebugParameter(str(joint_info[1]), joint_info[8], joint_info[9], joint_state[0])
    
    for i in range(12000):
        # if i == 499:
        #     pb.applyExternalForce(robot_id, -1, [10, 0, 0], [0, 0, 0], flags = pb.LINK_FRAME)
        #     print('Apply an external force!')

        # pb.setJointMotorControlArray(robot_id, [0, 1, 2, 3, 5, 6, 7, 8, 12], pb.POSITION_CONTROL, [0.0] * 9)
        # q_r_ankle = sensor_data['joint_pos'][4] + 0.1 * curr_com_pos[1]
        # q_l_ankle = sensor_data['joint_pos'][9] + 0.1 * curr_com_pos[0]
        # pb.setJointMotorControlArray(robot_id, [4, 9], pb.POSITION_CONTROL, [q_r_ankle, q_l_ankle])
        # q_abs_y = sensor_data['joint_pos'][10] + 0.1 * curr_com_pos[1]
        # q_abs_x = sensor_data['joint_pos'][11] + 0.1 * curr_com_pos[0]
        # pb.setJointMotorControlArray(robot_id, [10, 11], pb.POSITION_CONTROL, [q_abs_y, q_abs_x])

        

        for joint_idx, param_id in joint_debug_params.items():
            pb.setJointMotorControl2(robot_id, joint_idx, pb.POSITION_CONTROL, pb.readUserDebugParameter(param_id))

        sensor_data = get_sensor_data(robot_id, joint_indices)
        reaction_forces = get_ground_contact_forces(robot_id, plane_id)
        tau = map_contact_force_to_torque(robot_id, reaction_forces, sensor_data)
        N = get_inverse_dynamics_matrix(robot_id, sensor_data)

        com = get_com(robot_id)
        zmp = compute_zmp(reaction_forces)
        visualization(com_visual, com, zmp_visual, zmp)


        pb.setJointMotorControlArray(robot_id, list(range(6, 10)), pb.TORQUE_CONTROL, forces = (N[6:10]).tolist())
        pb.setJointMotorControlArray(robot_id, list(range(11, 15)), pb.TORQUE_CONTROL, forces = (N[11:15]).tolist())

        jpos = sensor_data['joint_pos']
        pb.setJointMotorControl2(robot_id, 10, pb.TORQUE_CONTROL, force = N[10] + -20 * (curr_com_pos[1] - 0)) #  -5 * jpos[10] 
        pb.setJointMotorControl2(robot_id, 15, pb.TORQUE_CONTROL, force = N[15] + -20 * (curr_com_pos[1] - 0)) #  + -5 * jpos[15]
        pb.setJointMotorControl2(robot_id, 16, pb.TORQUE_CONTROL, force = N[16] + -10 * (curr_com_pos[1] - 0))
        pb.setJointMotorControl2(robot_id, 17, pb.TORQUE_CONTROL, force = N[17] + -10 * (curr_com_pos[0] - 0))

        # tau = 0 ???
        # pb.setJointMotorControl2(robot_id, 10, pb.TORQUE_CONTROL, force = tau[10] + N[10])
        # pb.setJointMotorControl2(robot_id, 15, pb.TORQUE_CONTROL, force = tau[15] + N[15])
        # pb.setJointMotorControl2(robot_id, 16, pb.TORQUE_CONTROL, force = N[16])
        # pb.setJointMotorControl2(robot_id, 17, pb.TORQUE_CONTROL, force = N[17])

        pb.stepSimulation()

        # next_com_pos = get_com(robot_id)
        # com_vel = (next_com_pos - curr_com_pos) / (1 / 240.0)
        # curr_com_pos = next_com_pos

        time.sleep(0.01)

