import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-1)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,-0.17,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,-0.15,0.05])
robottId = p.loadURDF("D:/G/Poppy_data/urdf/poppy_ergo_jr_new.urdf",useFixedBase=True)
squareId = p.createCollisionShape(p.GEOM_BOX,radius=0.001, halfExtents = [0.01, 0.01,0.1])
l=p.createMultiBody(0,squareId,basePosition=cubeStartPos)
test=p.getBasePositionAndOrientation(squareId)
pbik = p.calculateInverseKinematics(robottId,7,cubeStartPos)

print(p.getNumJoints(robottId), " joints")
for j in range(p.getNumJoints(robottId)):
       print("%d: %s" % (j, p.getJointInfo(robottId, j))) # joint name
#print(boxId)
for i in range (10000):
    p.stepSimulation()
    if i>1000:
        m=p.setJointMotorControl2(robottId,0,p.POSITION_CONTROL,targetPosition=pbik[0])
        m = p.setJointMotorControl2(robottId, 1,p.POSITION_CONTROL, targetPosition=pbik[1])
        m = p.setJointMotorControl2(robottId, 2,p.POSITION_CONTROL, targetPosition=pbik[2])
        m = p.setJointMotorControl2(robottId, 3,p.POSITION_CONTROL, targetPosition=pbik[3])
        m = p.setJointMotorControl2(robottId, 4,p.POSITION_CONTROL, targetPosition=pbik[4])
        m = p.setJointMotorControl2(robottId, 6,p.POSITION_CONTROL, targetPosition=pbik[5])


    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robottId)
print(cubePos,cubeOrn)
p.disconnect()