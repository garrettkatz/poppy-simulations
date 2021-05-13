import numpy as np

# roll (x), pitch (y), yaw (z)
def RotX(theta):
    cost, sint = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, cost, -sint],
        [0, sint, cost]
    ])

def RotY(theta):
    cost, sint = np.cos(theta), np.sin(theta)
    return np.array([
        [cost, 0, sint],
        [0, 1, 0],
        [-sint, 0, cost]
    ])

def RotZ(theta):
    cost, sint = np.cos(theta), np.sin(theta)
    return np.array([
        [cost, -sint, 0],
        [sint, cost, 0],
        [0, 0, 1]
    ])

def GetRollPitchYawFromRot(R):
    pitch = np.arcsin(-R[2,0])
    yaw = np.arctan2(R[1,0], R[0,0])
    roll = np.arctan2(R[2,1],R[2,2])

    return np.array([roll, pitch, yaw])

def GetRotFromAngles(rpy):
    return RotZ(rpy[2]) @ RotY(rpy[1]) @ RotX(rpy[0])

def RotToTrans(R, p):
    return np.concatenate((np.concatenate((R, np.expand_dims(p, 1)), axis = 1), np.array([[0, 0, 0, 1]])))

def GetRotAndPosFromTrans(T):
    return T[:3, :3], T[:3, -1]

def SkewMatrix(k):
    return np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

def RodriguesFormula(k, theta):
    K = SkewMatrix(k)
    return np.eye(K.shape[0]) + np.sin(theta) * K + (1 - np.cos(theta)) * K.dot(K)

def degree2radian(d):
    return np.pi * d / 180

def radian2degree(r):
    return r * 180 / np.pi

def euler2quaternion(z, y, x, is_degree = True):
    if is_degree:
        z = degree2radian(z)
        y = degree2radian(y)
        x = degree2radian(x)
    else:
        assert z >= -1 and y >= -1 and x >= -1 and z <= 1 and y <= 1 and x <= 1
    cosz2, sinz2 = np.cos(z/2), np.sin(z/2)
    cosy2, siny2 = np.cos(y/2), np.sin(y/2)
    cosx2, sinx2 = np.cos(x/2), np.sin(x/2)
    return (
        cosx2 * cosy2 * cosz2 + sinx2 * siny2 * sinz2,
        sinx2 * cosy2 * cosz2 - cosx2 * siny2 * sinz2,
        cosx2 * siny2 * cosz2 + sinx2 * cosy2 * sinz2,
        cosx2 * cosy2 * sinz2 - sinx2 * siny2 * cosz2
    )

def quaternion2euler(quaternion):
    # length(quaternion) = 1
    qr, qi, qj, qk = quaternion
    return np.array([
        [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)],
        [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],
        [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2)]
    ])