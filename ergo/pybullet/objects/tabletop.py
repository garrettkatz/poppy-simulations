import os, sys
import pybullet as pb
import math
def add_cube(position, half_extents, rgb, mass=0.1):

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
# Updated upstream

def add_rndm_Obj(boxes, mass= 0.1):
    mid =0
    cids,vids= [], []
    pos, ext, rgb = zip(*boxes)
    for b in range(len(boxes)):
        cids.append(pb.createCollisionShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=ext[b],
        ))
        vids.append(pb.createVisualShape(
            shapeType=pb.GEOM_BOX,
            halfExtents=ext[b],
            rgbaColor=rgb[b] + (1,),  # alpha is opaque
        ))
    mid = pb.createMultiBody(
        baseMass=mass,
        linkMasses=[1] * len(boxes),
        linkCollisionShapeIndices=cids,
        linkVisualShapeIndices=vids,
        linkPositions=pos,
        linkOrientations=[(0, 0, 0, 1)] * len(boxes),
        linkInertialFramePositions=pos,
        linkInertialFrameOrientations=[(0, 0, 0, 1)] * len(boxes),
        linkParentIndices=[0] * len(boxes),
        linkJointTypes=[pb.JOINT_FIXED] * len(boxes),
        linkJointAxis=[(0, 0, 0, 1)] * len(boxes),
    )

    return mid
    return mid

def add_Obj_compound(position,half_extents,rgb,mass = 0.01):
    cid = pb.createCollisionShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=half_extents,
    )
    cid2 = pb.createCollisionShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=[half_extents[0],half_extents[1],0.001],
    )
    vid = pb.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgb + (1,),  # alpha is opaque
    )
    vid2 = pb.createVisualShape(shapeType = pb.GEOM_BOX,
                                halfExtents=[half_extents[0],half_extents[1],0.001],
                                rgbaColor = (0,1,1)+(1,),
                                #visualFramePosition=[position[0],position[1],position[2]-half_extents[2]],
                                )
    obj = pb.createMultiBody(
        baseMass=mass,
        basePosition=position,
        #baseMass=mass,
        #baseCollisionShapeIndex=cid,
        #baseVisualShapeIndex=vid,
        linkMasses=[0.01,0.001],
        linkCollisionShapeIndices=[cid,cid2],
        linkVisualShapeIndices=[vid,vid2],
        linkPositions=[[0,0,0],[0,0,0-half_extents[2]]],
        linkOrientations=[(0, 0, 0, 1),(0, 0, 0, 1)],
        linkInertialFramePositions=[[0,0,0],[0,0,0]],
        linkInertialFrameOrientations=[(0, 0, 0, 1),(0, 0, 0, 1)],
        linkParentIndices=[0,0] ,
        linkJointTypes=[pb.JOINT_FIXED,pb.JOINT_FIXED],
        linkJointAxis=[(0, 0, 0, 1),(0, 0, 0, 1)] ,
    )

    return obj
def add_box_compound(boxes, mass=0.001):

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

#
def add_obj(position , urdf_path ,orn):
    loader = pb.loadURDF(urdf_path,position,orn)
    return loader
# Stashed changes
def add_table(mass=0):
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

