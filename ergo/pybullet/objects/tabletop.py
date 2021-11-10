import os, sys
import pybullet as pb

def add_cube(position, half_extents, rgb):

    cid = pb.createCollisionShape(
        shapeType = pb.GEOM_BOX,
        halfExtents = half_extents,
    )
    vid = pb.createVisualShape(
        shapeType = pb.GEOM_BOX,
        halfExtents = half_extents,
        rgbaColor = rgb+(1,), # alpha is opaque
    )
    mid = pb.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition = position,
    )    
    return mid

def add_table():
    position = (0, -.5, .2)
    half_extents = (.5, .2, .2)
    rgb = (.5, .5, .5)
    mid = add_cube(position, half_extents, rgb)
    return mid

if __name__ == "__main__":

    sys.path.append(os.path.join('..', 'envs'))
    from ergo import PoppyErgoEnv

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL)

    mid = add_table()

    input("..")

