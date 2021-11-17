import os, sys
import pybullet as pb

def add_cube(position, half_extents, rgb,mass=1):

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
        baseMass=mass,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition = position,
    )

    return mid

def add_table(mass=100):
    position = (0, -.4, .2)
    half_extents = (.5, .2, .2)
    rgb = (.5, .5, .5)
    mid = add_cube(position, half_extents, rgb,mass)
    return mid

if __name__ == "__main__":

    sys.path.append(os.path.join('..', 'envs'))
    from ergo import PoppyErgoEnv

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL)

    mid = add_table()

    input("..")

