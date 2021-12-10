import os, sys
import pybullet as pb

def add_cube(position, half_extents, rgb, mass=1):

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

def add_box_compound(boxes, mass=1):

    pos, ext, rgb = zip(*boxes)
    cids, vids = [], []

    for b in range(len(boxes)):
        cids.append(pb.createCollisionShape(
            shapeType = pb.GEOM_BOX,
            halfExtents = ext[b],
        ))
        vids.append(pb.createVisualShape(
            shapeType = pb.GEOM_BOX,
            halfExtents = ext[b],
            rgbaColor = rgb[b]+(1,), # alpha is opaque
        ))

    mid = pb.createMultiBody(
        baseMass=mass,
        linkMasses=[1]*len(boxes),
        linkCollisionShapeIndices=cids,
        linkVisualShapeIndices=vids,
        linkPositions=pos,
        linkOrientations=[(0,0,0,1)]*len(boxes),
        linkInertialFramePositions=pos,
        linkInertialFrameOrientations=[(0,0,0,1)]*len(boxes),
        linkParentIndices=[0]*len(boxes),
        linkJointTypes=[pb.JOINT_FIXED]*len(boxes),
        linkJointAxis=[(0,0,0,1)]*len(boxes),
    )

    return mid

def table_position():
    return (0, -.4, .2)

def table_half_extents():
    return (.5, .2, .2)

def add_table(mass=100):
    position = table_position()
    half_extents = table_half_extents()
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

