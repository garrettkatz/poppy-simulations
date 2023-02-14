import pybullet as pb
import os
import numpy as np
from poppy_env import PoppyEnv

class PoppyErgoEnv(PoppyEnv):
    
    # Ergo-specific urdf loading logic
    def load_urdf(self, use_fixed_base, use_self_collision):
        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/ergo'
        pb.setAdditionalSearchPath(fpath)
        robot_id = pb.loadURDF(
            'poppy_ergo.pybullet.urdf',
            basePosition = (0, 0, .43),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            useFixedBase=use_fixed_base,
            flags = pb.URDF_USE_SELF_COLLISION if use_self_collision else 0,
        )
        return robot_id

    # Get mirrored version of position across left/right halves of body
    def mirror_position(self, position):
        mirrored = np.empty(len(position))
        for i, name in self.joint_name.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[self.joint_index[mirror_name]] = position[i] * sign        
        return mirrored        

    # Get image from head camera
    def get_camera_image(self):

        # Get current pose of head camera
        # link index should be same as parent joint index?
        state = pb.getLinkState(self.robot_id, self.joint_index["head_cam"])
        pos, quat = state[:2]
        M = np.array(pb.getMatrixFromQuaternion(quat)).reshape((3,3)) # local z-axis is third column

        # Calculate camera target and up vector
        camera_position = tuple(p + d for (p,d) in zip(pos, .1*M[:,2]))
        target_position = tuple(p + d for (p,d) in zip(pos, .4*M[:,2]))
        up_vector = tuple(M[:,1])
        
        # Capture image
        width, height = 128, 128
        # width, height = 8, 8 # doesn't actually make much speed difference
        view = pb.computeViewMatrix(
            cameraEyePosition = camera_position,
            cameraTargetPosition = target_position, # focal point
            cameraUpVector = up_vector,
        )
        proj = pb.computeProjectionMatrixFOV(
            # fov = 135,
            fov = 90,
            aspect = height/width,
            nearVal = 0.01,
            farVal = 2.0,
        )
        # rgba shape is (height, width, 4)
        _, _, rgba, _, _ = pb.getCameraImage(
            width, height, view, proj,
            flags = pb.ER_NO_SEGMENTATION_MASK) # not much speed difference
        # rgba = np.empty((height, width, 4)) # much faster than pb.getCameraImage
        return rgba, view, proj

    # convert angle array to pypot-compatible dictionary
    def angle_dict(self, angle_array, convert=True):
        pypot_motors = [
            'abs_y', 'abs_x', 'abs_z', 'bust_y', 'bust_x', 'head_z', 'head_y',
            'l_shoulder_y', 'l_shoulder_x', 'l_arm_z', 'l_elbow_y', 'r_shoulder_y',
            'r_shoulder_x', 'r_arm_z', 'r_elbow_y', 'l_hip_x', 'l_hip_z', 'l_hip_y',
            'l_knee_y', 'l_ankle_y', 'r_hip_x', 'r_hip_z', 'r_hip_y', 'r_knee_y', 'r_ankle_y']
        return {
            name: angle_array[self.joint_index[name]] * (180/np.pi if convert else 1)
            for name in pypot_motors}
            
# convert from physical robot angles to pybullet angles
# angles[name]: angle for named joint (in degrees)
# degrees are converted to radians
def convert_angles(angles):
    cleaned = {}
    for m,p in angles.items():
        cleaned[m] = p * np.pi / 180
    return cleaned

if __name__ == "__main__":
    
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True, show=True)

    # confirm that poppy urdf faces the -y direction
    pos, orn, vel, ang = env.get_base()
    env.set_base((pos[0], -1, pos[2]), orn, vel, ang)

    # got from running camera.py
    cam = (1.200002670288086,
        15.999960899353027,
        -31.799997329711914,
        (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
    pb.resetDebugVisualizerCamera(*cam)

    action = env.get_position()

    # avoid self-collision
    action[env.joint_index["r_shoulder_x"]] -= .2
    action[env.joint_index["l_shoulder_x"]] += .2
    env.set_position(action)

    print(action)
    print(env.joint_low)
    print(env.joint_high)

    i = env.joint_index["r_shoulder_x"]
    print(i, env.joint_low[i], env.joint_high[i])
    action[i] = .9 * env.joint_low[i]

    input('...')

    # confirm that joint angle limits are enforced, though apparently self-collisions are not
    # action[0] = 10 * env.joint_high[0]
    # action[0] = 10 * env.joint_low[0]

    env.goto_position(action)

    input('...')



