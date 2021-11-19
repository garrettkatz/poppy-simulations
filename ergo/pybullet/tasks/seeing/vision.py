import numpy as np
import pybullet as pb
import matplotlib.pyplot as pt

def get_object_poses(env):
    # take the picture
    rgba, view, proj = env.get_camera_image()

    # replace this with NN
    objs = [
        pb.getBasePositionAndOrientation(bid)
        for bid in range(3, pb.getNumBodies()) # first two bodies are robot and table
    ]
    
    return objs

def to_image_coordinates(rgba, view, proj, x):
    # rgba, view, proj: as returned by get_camera_image
    # x: 3xN array of N 3d points to transform

    # image width and height
    width, height = rgba.shape[1], rgba.shape[0]

    # view and projection matrix transforms
    view = np.array(view).reshape((4,4)).T
    proj = np.array(proj).reshape((4,4)).T

    # homogenous coordinates
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)

    # clipping space coordinates
    x = proj @ view @ x

    # perspective projection
    x = np.stack((x[0]/x[3], x[1]/x[3]))

    # image coordinates
    ij = np.stack(((1-x[1])*height, (1+x[0])*width))/2

    return ij

def tweak_grip(env, targets):

    # take the picture
    rgba, view, proj = env.get_camera_image()

    # get focal point for sub-image around targets
    x = np.array(targets).mean(axis=0).reshape((3, 1))

    # clip sub-image
    ij = to_image_coordinates(rgba, view, proj, x)
    i, j = tuple(ij.astype(int).flatten())
    print(i, j)
    print(rgba.shape)
    sub_image = rgba[i-5:i+5, j-5:j+5,:]
    
    # show sub-image
    pt.imshow(sub_image)
    pt.show()

    # replace this with NN(sub_image) that modifies targets
    return targets

