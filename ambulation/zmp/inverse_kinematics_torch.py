import torch
import numpy as np

def get_rotation_mat(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_roll = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    R_pitch = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    R_yaw = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return R_roll @ R_pitch @ R_yaw

def get_position_vec(xyz):
    return np.expand_dims(np.append(xyz, 1), axis = 1)

def get_transform_mat(xyz, rpy):
    R = get_rotation_mat(rpy)
    p = get_position_vec(xyz)
    return np.hstack((np.vstack((R, np.array([0, 0, 0]))), p))





def skew_matrix(k):
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

def rodrigues_rotation_formula(theta, axis):
    K = skew_matrix(axis)
    return np.eye(K.shape[0]) + np.sin(theta) * K + (1 - np.cos(theta)) * K.dot(K)

def forward_kinematics(theta, axis, last_R, offset, last_pos):
    p = last_R @ offset + last_pos
    R = last_R @ rodrigues_rotation_formula(theta, axis)
    return p, R

if __name__ == '__main__':
    pass