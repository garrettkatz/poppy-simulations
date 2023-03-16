import pybullet as pb

import pybullet_data

client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0,0,-10)
planeId = pb.loadURDF("plane.urdf")

cid = pb.createCollisionShape(pb.GEOM_SPHERE, radius=.25)
vid = pb.createVisualShape(pb.GEOM_SPHERE, radius=.25)

mid = pb.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=cid,
    baseVisualShapeIndex=vid,
    baseOrientation=(0,0,1,0),
    linkMasses = [1],
    linkCollisionShapeIndices=[cid],
    linkVisualShapeIndices=[vid],
    linkPositions=[(0,0,1)],
    linkOrientations=[(.7071,0,.7071,0)],
    linkInertialFramePositions=[(0,0,1)],
    linkInertialFrameOrientations=[(0,0,0,1)],
    linkParentIndices=[0],
    linkJointTypes=[pb.JOINT_REVOLUTE],
    linkJointAxis=[(1,0,0,0)],
)

pb.resetJointState(mid, 0, 1.)

# for _ in range(100):
#     input('.')
while True:
    pb.stepSimulation()

pb.disconnect()

