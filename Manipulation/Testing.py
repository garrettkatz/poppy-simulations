import pybullet as p
import time
import pybullet_data
from scipy.spatial.transform import Rotation
import numpy as np
_CUBE_WIDTH = 0.02

_cube_corners = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ]
) * (_CUBE_WIDTH / 2)
def get_cube_corner_positions(pose):
    """Get the positions of the cube's corners with the given pose.
    Args:
        pose (Pose):  Pose of the cube.
    Returns:
        (array, shape=(8, 3)): Positions of the corners of the cube in the
            given pose.
    """
    rotation = Rotation.from_quat(pose.orientation)
    translation = np.asarray(pose.position)

    return rotation.apply(_cube_corners) + translation

def get_intermediate_point(cubepos , robotpos):
    res_list = []
    for i in range(0, len(cubepos)):
        res_list.append((cubepos[i] + robotpos[i])/2)
    return res_list
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,-0.17,0]
intermediate_point_1 = [0,-0.17,0.1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
waypoints = []
waypoints_tips =[]
robottId = p.loadURDF("poppy_ergo_jr_new.urdf",useFixedBase=True)
robottId2 = p.loadURDF("Cube.urdf",cubeStartPos)
intermediate_point_2 = [0,-0.17,0]
intermediate_point_3 = [0.2,-0.10,0.3]
cubeStartPos,cubeStartOrientation = p.getBasePositionAndOrientation(robottId2)
cubeStartPos1,cubeStartOrientation1 = p.getBasePositionAndOrientation(robottId)
print(cubeStartPos1,cubeStartOrientation1)
numJoints = p.getNumJoints(robottId)
def accurateCalculateInverseKinematics(kukaId, endEffectorId, targetPos, threshold, maxIter):
  closeEnough = False
  iter = 0
  dist2 = 1e30
  data = p.getJointStates(kukaId, [0,1,2,3,4])
  while (not closeEnough and iter < maxIter):
    jointPoses = p.calculateInverseKinematics(kukaId, 7, targetPos)
    for i in range(5):
      p.resetJointState(kukaId, i, jointPoses[i])
    ls = p.getLinkState(kukaId, 7)
    newPos = ls[4]
    diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
    dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    closeEnough = (dist2 < threshold)
    iter = iter + 1
  #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))    for i in range(5):
  for i in range(5):
    p.resetJointState(kukaId, i, data[i][0])
  return jointPoses
#pbik = p.calculateInverseKinematics(robottId,7,intermediate_point_1)
robot_join_pose = p.getLinkState(robottId,7)

pbik = p.calculateInverseKinematics(robottId,7,intermediate_point_1)

pbik_1 = p.calculateInverseKinematics(robottId,7,intermediate_point_2)
pbik_2 = p.calculateInverseKinematics(robottId,7,intermediate_point_3)
waypoints.append(pbik)
waypoints.append(pbik_1)
waypoints.append(pbik_2)
waypoints_tips.append(0.5)
waypoints_tips.append(-0.4)
waypoints_tips.append(0.5)
print(p.getNumJoints(robottId), " joints")
for j in range(p.getNumJoints(robottId)):
       print("%d: %s" % (j, p.getJointState(robottId, j))) # joint name
#print(boxId)
while len(waypoints)>0:

    for i in range(1000):
        p.stepSimulation()
        if i>100 and i <900:
            m = p.setJointMotorControl2(robottId, 0, p.POSITION_CONTROL, targetPosition=waypoints[0][0], maxVelocity=3)
            m = p.setJointMotorControl2(robottId, 1, p.POSITION_CONTROL, targetPosition=waypoints[0][1], maxVelocity=3)
            m = p.setJointMotorControl2(robottId, 2, p.POSITION_CONTROL, targetPosition=waypoints[0][2], maxVelocity=3)
            m = p.setJointMotorControl2(robottId, 3, p.POSITION_CONTROL, targetPosition=waypoints[0][3], maxVelocity=3)
            m = p.setJointMotorControl2(robottId, 4, p.POSITION_CONTROL, targetPosition=waypoints[0][4], maxVelocity=3)
        print(p.getLinkState(robottId, 7))
        if i>900:
            m = p.setJointMotorControl2(robottId, 6, p.POSITION_CONTROL, targetPosition=waypoints_tips[0])
        time.sleep(1. / 240.)
    waypoints.pop(0)
    waypoints_tips.pop(0)
    print(p.getLinkState(robottId,7))
#for i in range (10000):
 #   p.stepSimulation()
    #("stable:", p.getBasePositionAndOrientation(robottId2))
   # print("\n")
  #  if i>200 and i<800:
   #     print("\ni=1000\n")
    #    m = p.setJointMotorControl2(robottId, 0, p.POSITION_CONTROL, targetPosition=pbik[0], maxVelocity=3)
     #   m = p.setJointMotorControl2(robottId, 1,p.POSITION_CONTROL, targetPosition=pbik[1], maxVelocity=3)
      #  m = p.setJointMotorControl2(robottId, 2,p.POSITION_CONTROL, targetPosition=pbik[2], maxVelocity=3)
       # m = p.setJointMotorControl2(robottId, 3,p.POSITION_CONTROL, targetPosition=pbik[3], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 4,p.POSITION_CONTROL, targetPosition=pbik[4], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 6,p.POSITION_CONTROL, targetPosition=pbik[5])
        #print("moving:", p.getBasePositionAndOrientation(robottId2))
    #if i>500 and i <1000:
     #   print("\ni=2000\n")
      #  m = p.setJointMotorControl2(robottId, 6, p.POSITION_CONTROL, targetPosition=0.5)
    #if i>800 and i<1200:
     #   print("\ni=1000\n")
      #  m = p.setJointMotorControl2(robottId, 0, p.POSITION_CONTROL, targetPosition=pbik_1[0], maxVelocity=3)
       # m = p.setJointMotorControl2(robottId, 1,p.POSITION_CONTROL, targetPosition=pbik_1[1], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 2,p.POSITION_CONTROL, targetPosition=pbik_1[2], maxVelocity=3)
        #m# = p.setJointMotorControl2(robottId, 3,p.POSITION_CONTROL, targetPosition=pbik_1[3], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 4,p.POSITION_CONTROL, targetPosition=pbik_1[4], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 6,p.POSITION_CONTROL, targetPosition=pbik[5])
        #print("moving:", p.getBasePositionAndOrientation(robottId2))
    #if i>1000:
     #   m = p.setJointMotorControl2(robottId, 6, p.POSITION_CONTROL, targetPosition=-0.4)
    #if i>1200:
        #m = p.setJointMotorControl2(robottId, 0, p.POSITION_CONTROL, targetPosition=pbik_2[0], maxVelocity=3)
        #m = p.setJointMotorControl2(robottId, 1, p.POSITION_CONTROL, targetPosition=pbik_2[1], maxVelocity=3)
       # m = p.setJointMotorControl2(robottId, 2, p.POSITION_CONTROL, targetPosition=pbik_2[2], maxVelocity=3)
      #  m = p.setJointMotorControl2(robottId, 3, p.POSITION_CONTROL, targetPosition=pbik_2[3], maxVelocity=3)
     #   m = p.setJointMotorControl2(robottId, 4, p.POSITION_CONTROL, targetPosition=pbik_2[4], maxVelocity=3)
    #time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robottId)

print(cubePos,cubeOrn)

p.disconnect()

