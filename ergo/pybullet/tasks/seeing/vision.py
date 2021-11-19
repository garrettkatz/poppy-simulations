import pybullet as pb

def get_object_poses(env):
    # take the picture
    rgba, view, proj = env.get_camera_image()

    # replace this with NN
    objs = [
        pb.getBasePositionAndOrientation(bid)
        for bid in range(3, pb.getNumBodies()) # first two bodies are robot and table
    ]
    
    return objs

def tweak_grip(env, targets):
    
    # replace this with NN that modifies targets
    return targets

