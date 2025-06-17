import copy
import os, sys
import numpy as np
import pybullet as pb
import math
import glob
from xml.etree.ElementTree import Element,tostring
#import tabletop as tt
#from dicttoxml import dicttoxml
#import dicttoxml
import xmltodict

sys.path.append(os.path.join('..', '..', 'envs'))
from ergo import PoppyErgoEnv
import motor_control2 as mc

sys.path.append(os.path.join('..', '..', 'objects'))
from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
def dict_to_xml(tag, d):

    elem = Element(tag)
    for key, val in d.items():

     child = Element(key)
     test1 = str(val)
     child.text = test1
    elem.append(child)

    return elem


class Data_Dict:
    def __init__(self):
        self.Dict_result = {}
        self.Dict_positions = {}
        self.Dict_extents = {}


    def AddObjtoDict(self,objId,positions,extents,parts):

        self.Dict_result[objId] = np.zeros(parts+1)
        self.Dict_positions[objId] = positions
        self.Dict_extents[objId] = extents

    def Addresult(self,ObjId,result):
        self.Dict_result[ObjId] = result


def get_tip_targets(p, q, d):
    m = q
    t1 = p[0]-d*m[0], p[1]-d*m[3], p[2]-d*m[6]
    t2 = p[0]+d*m[0], p[1]+d*m[3], p[2]+d*m[6]
    return (t2, t1)
def get_tip_targets2(p, q, d):
    m = q
    t1 = p[0]-d*m[1], p[1]-d*m[4], p[2]-d*m[7]
    t2 = p[0]+d*m[1], p[1]+d*m[4], p[2]+d*m[7]
    return (t1, t2)


def slot_arrays_mod(inn, out):
    # inn/out: inner/outer half extents, as np arrays
    mid = (inn + out) / 2
    ext = (out - inn) / 2
    positions = np.array([
        [-mid[0], 0, 0],
        [0, out[1] * np.random.uniform(-1, 1), out[2] * np.random.uniform(-1, 1)],
        [0, -out[1] * np.random.uniform(-1, 1), -out[2] * np.random.uniform(-1, 1)],

    ])
    extents = [
        (ext[0], out[1], out[2]),
        (inn[0] - np.random.uniform(-1, 1) * (inn[0]) / 2, out[2], ext[1]),
        (inn[0], out[2], ext[1]),

    ]
    return positions, extents


dim = np.array([.01, .01, .01])


def ObjectOpenPositions(positions, extents):
    voxel_dim = extents[0]
    voxel_positions = positions.copy()
    # add to voxel position
    for pos in positions:
        for i in range(3):  # xyz
            base = pos
            new_base2 = base.copy()
            new_base = base.copy()
            new_base[i] = base[i] - 2 * dim[i]
            if new_base not in positions:
                voxel_positions.append(new_base)

            new_base2[i] = base[i] + 2 * dim[i]
            if new_base2 not in positions:
                voxel_positions.awwwppend(new_base2)
    return extents, voxel_positions


def mutate(Obj_extents, Obj_pos, Voxel_pos, Voxel_extents):
    new_mutant_obj_pos = Obj_pos.copy()
    print(new_mutant_obj_pos[len(new_mutant_obj_pos) - 1])
    print(np.random.randint(0, len(voxel_pos) - 1))
    print(Voxel_pos[np.random.randint(0, len(voxel_pos) - 1)])
    new_mutant_obj_pos[len(new_mutant_obj_pos) - 1] = Voxel_pos[
        np.random.randint(0, len(voxel_pos) - 1)]  # worst location on fitness function
    return new_mutant_obj_pos, Obj_extents


def Actual_voxels():
    # combine voxels(0), postions(1)
    return 0


def Generative_Shapes(dim, no_parts, base_pos):
    new_base = base_pos
    positions = []
    positions.append(new_base.copy())
    extents = []
    extents.append(dim)
    while len(positions) < no_parts:
        # for i in range(no_parts-1):
        old_base = new_base
        rng_xyz = np.random.randint(3)
        rng_pos_neg = np.random.randint(2)
        if rng_pos_neg == 0:
            new_base[rng_xyz] = old_base[rng_xyz] - 2 * dim[rng_xyz]
        else:
            new_base[rng_xyz] = old_base[rng_xyz] + 2 * dim[rng_xyz]
        if new_base not in positions:
            positions.append(new_base.copy())
        else:
            continue
        extents.append(dim)
    maxz = np.amin(positions)
    return positions, extents, maxz


dim = np.array([.01, .01, .01])


# savedstate = pb.saveState(env.)

# --
class Obj:
    def __init__(self,extents,No_parts,rbg):
        self.positions = []
        self.extents = extents
        self.dim = extents
        self.isMutant = False
        self.ObjId = -99
        self.basePosition = [0,0,0]
        #self.objId = -99
        #or self.mutants = list of mutants
        self.ParentId = -1
        self.voxels = np.zeros((No_parts+2,)*3)

        self.PositionsAvailable = [] #positions available for mutation
        self.NoOfParts = No_parts
        self.rgb = rbg
        self.maxz = 0

    def GenerateObject(self,dim,NoOfParts,base_pos):
        new_base = base_pos
        positions = []
        positions.append(new_base.copy())
        extents = []
        extents.append(dim)
        while len(positions) < NoOfParts:
            # for i in range(no_parts-1):
            #temp = old_base
            old_base = new_base
            rng_xyz = np.random.randint(3)
            rng_pos_neg = np.random.randint(2)
            if rng_pos_neg == 0:
                new_base[rng_xyz] = old_base[rng_xyz] - 2 * dim[rng_xyz]
            else:
                new_base[rng_xyz] = old_base[rng_xyz] + 2 * dim[rng_xyz]
            if new_base not in positions:
                positions.append(new_base.copy())
            else:
                new_base = old_base
                continue
            extents.append(dim)
        maxz = np.amin(positions)
        self.positions =positions
        self.extents = extents
        self.maxz = maxz
        return self
    def Generate_Voxels(self):

        for pos in self.positions:
            self.voxels[pos] = 1
        return self

    def MutateObject(self):
        new_mutant_obj_pos = self.positions.copy()
        #print(new_mutant_obj_pos[len(new_mutant_obj_pos) - 1])
        #print(np.random.randint(0, len(voxel_pos) - 1))
        #print(self.PositionsAvailable[np.random.randint(0, len(voxel_pos) - 1)])
        obj = self.GetOpenPosition()
        #self.PositionsAvailable = obj.PositionsAvailable

        new_mutant_obj_pos[len(new_mutant_obj_pos) - 1] = self.PositionsAvailable[
            np.random.randint(0, len(self.PositionsAvailable) - 1)]  # worst location on fitness function
        newobj = Obj(self.extents,self.NoOfParts,self.rgb)
        newobj.positions = new_mutant_obj_pos
        newobj.isMutant= True
        newobj.ParentId = self.ObjId
        return newobj

    def clone_and_mutate_to(self, new_position):
        new_obj = Obj(self.extents, self.NoOfParts, self.rgb)
        new_obj.positions = [pos.copy() for pos in self.positions]
        new_obj.positions[-1] = new_position.copy()
        new_obj.isMutant = True
        new_obj.ParentId = getattr(self, "ObjId", None)
        new_obj.dim = self.dim.copy()
        return new_obj
    def MutateObject_RL(self,voxel_,face_):
        new_mutant_obj_pos = self.positions.copy()
        obj = self.GetOpenPosition()

        new_mutant_obj_pos[len(new_mutant_obj_pos) - 1] = self.PositionsAvailable[
            np.random.randint(0, len(self.PositionsAvailable) - 1)]  # worst location on fitness function
        newobj = Obj(self.extents,self.NoOfParts,self.rgb)
        newobj.positions = new_mutant_obj_pos
        newobj.isMutant= True
        newobj.ParentId = self.ObjId
        return newobj

    def is_connected(self,positions):
        if len(positions) <= 1:
            return True

        pos_set = {tuple(pos) for pos in positions}
        visited = set()
        stack = [next(iter(pos_set))]

        neighbor_offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if len(visited) == len(pos_set):  # Early exit
                return True
            for dx, dy, dz in neighbor_offsets:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if neighbor in pos_set and neighbor not in visited:
                    stack.append(neighbor)

        return False  # Not all positions were visited

    def Multiple_MutateObject(self):
        self.GetOpenPosition()  # Updates self.PositionsAvailable
        MutantList = []

        for pos in self.PositionsAvailable:
            # Create new position list with the last part replaced
            new_mutant_obj_pos = self.positions[:-1] + [pos.copy()]

            # Create the mutant object
            newobj = Obj(self.extents, self.NoOfParts, self.rgb)
            newobj.positions = new_mutant_obj_pos
            newobj.isMutant = True
            newobj.ParentId = self.ObjId
            newobj.dim = self.dim.copy()

            MutantList.append(newobj)

        return MutantList

    def _SpawnObject(self,env,dim,noofparts,basepos):
        self.GenerateObject(dim,noofparts,basepos)
        Boxes = list(zip(map(tuple, self.positions), self.extents, self.rgb))
        ObjInfo = add_box_compound(Boxes)
        return ObjInfo

    def crossover(self, obj1, obj2, npartsA):

        ChildObj = Obj(obj1.extents, obj1.NoOfParts, obj1.rgb)
        ChildObj.isMutant = True
        ChildObj.ParentId = obj1.ObjId
        ChildObj.dim = obj1.dim.copy()
        dummy_obj2pos = obj2.positions.copy()
        childpos = list()
        for i in range(len(obj1.positions)):
            if i <= npartsA:
                childpos.append(obj1.positions[i].copy())
            else:
                temp_pos_list = list()
                x = obj2.positions[i][0] - obj2.positions[npartsA][0] + childpos[npartsA][0]
                y = obj2.positions[i][1] - obj2.positions[npartsA][1] + childpos[npartsA][1]
                z = obj2.positions[i][2] - obj2.positions[npartsA][2] + childpos[npartsA][2]
                temp_pos_list.append(x)
                temp_pos_list.append(y)
                temp_pos_list.append(z)

                childpos.append(temp_pos_list.copy())
                temp_pos_list.clear()

        ChildObj.positions = childpos
        return ChildObj

    def get_open_positions(self,positions, occupied=None):
        """
        Find open voxel positions adjacent to the current object blocks.

        Args:
            positions: List of current block positions (List[List[int]]).
            occupied: Optional set of positions to consider occupied (set of tuples).
                      If None, will use positions as the only occupied space.

        Returns:
            List of available open positions (List[List[int]]).
        """
        neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        pos_set = set(tuple(p) for p in positions)
        occupied = occupied or pos_set

        open_set = set()

        for pos in positions:
            x, y, z = pos
            for dx, dy, dz in neighbors:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor not in occupied:
                    open_set.add(neighbor)

        return [list(pos) for pos in open_set]
    def GetOpenPosition(self):
        positions = self.positions.copy()
        extents = self.extents
        voxel_dim = extents[0]
        voxel_positions = []
        # add to voxel position
        for pos in positions:
            for i in range(3):  # xyz
                base = pos
                new_base2 = base.copy()
                new_base = base.copy()
                new_base[i] = base[i] - 2 * dim[i]
                if new_base not in positions:
                    voxel_positions.append(new_base)

                new_base2[i] = base[i] + 2 * dim[i]
                if new_base2 not in positions:
                    voxel_positions.append(new_base2)
        self.PositionsAvailable = voxel_positions.copy()
        return self

class experiment:
    def __init__(self):

        self.env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)
        sys.path.append(os.path.join('..', '..', 'envs'))
        sys.path.append(os.path.join('..', '..', 'objects'))
        add_table()
        self.t_pos = table_position()
        self.t_ext = table_half_extents()
        self.init_robot_pos = []

    def CreateScene(self):

        #s_pos = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - self.maxz)
        #s_pos2 = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - self.maxz) + np.random.randn(
         #   3) * np.array([0.5, 0, 0])
        angles = self.env.angle_dict(self.env.get_position())
        angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
        self.init_robot_pos = self.env.angle_array(angles).copy()
        self.env.set_position(self.env.angle_array(angles))
        return 0
    def reset_robot(self):
        self.env.set_position(self.init_robot_pos)
        return 0

    def Spawn_Object(self,obj,b_position=0,orn =(1.0, 1.0, 0.0, 1)):
        Boxes = list(zip(map(tuple, obj.positions), obj.extents, obj.rgb))
        ObjInfo = add_box_compound(Boxes)
        a=self.t_pos[0]
        b=self.t_pos[1] + self.t_ext[1] / 2
        c=self.t_pos[2] + self.t_ext[2] + obj.dim[2] / 2 - obj.maxz -0.001
        b_position = (self.t_pos[0] +0.02, self.t_pos[1] + self.t_ext[1] / 2, self.t_pos[2] + self.t_ext[2] + obj.dim[2] / 2 - obj.maxz +0.002)
        pb.resetBasePositionAndOrientation(ObjInfo, b_position, orn) # use orn to change orientation
        obj.basePosition = b_position
        obj.ObjId = ObjInfo
        return ObjInfo

    def MoveToPos(self,pos,opening_width,arm="right"):
        target_pos = copy.deepcopy(pos)
        #target_pos[2] = target_pos[2]
        quat = pb.getMatrixFromQuaternion([0,0,0,1])
        tarargs = get_tip_targets(target_pos, quat, opening_width)
        i_k = mc.balanced_reach_ik(self.env, tarargs, arm)
        self.env.goto_position(i_k, 1)
        print("Moving-1")
        return 1

    def Tip_Collision_detect(self):
        return 0
    def MoveThroughPoints(self,Targets):

        for tar in Targets:
            print(tar)
        return 1

dims = np.array([.01, .01, .01])
n_parts = 6
rgb = [(.75, .25, .25)] * n_parts
obj = Obj(dims,n_parts,rgb)
obj = obj.GenerateObject(dims,n_parts,[0,0,0])
sys.path.append(os.path.join('..', '..', 'envs'))


# sys.path.append(os.path.join('..', '..', 'objects'))
# from tabletop import add_table, add_cube, add_obj, add_box_compound, table_position, table_half_extents
# #env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)
# exp_obj = Obj(dims,n_parts,rgb)
# exp_obj = exp_obj.GenerateObject(dims,n_parts,[0,0,0])
# xprmt = experiment()
# xprmt = xprmt.SpawnObject(exp_obj)
#
#
# if __name__ == "__main__":
#     sys.path.append(os.path.join('..', '..', 'envs'))
#     from ergo import PoppyErgoEnv
#     import motor_control2 as mc
#     sys.path.append(os.path.join('..', '..', 'objects'))
#     from tabletop import add_table, add_cube,add_obj,add_box_compound,table_position,table_half_extents
#
#     # this launches the simulator
#     # fix the base to avoid balance issues : Removed in Update
#     env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)
#     # this adds the table
#     add_table()
#
#     # add a few cubes at random positions on the table
# Error = []
# Error_count=[]
# urdfList = []
# fpath = "D:/G/Poppy_data/**/*.urdf"
# files = glob.glob(fpath,recursive=True)
# test = np.random.randn()


#
#
# Dataset = Data_Dict()
#
# for loop_counter in range(100):
#     n_parts = 6
#     pos, ext, maxz = Generative_Shapes(dim, n_parts, [0, 0, 0])
#     voxel_extents,voxel_pos = ObjectOpenPositions(pos,ext)
#     dims = np.array([.01, .01, .01])
#     mut_pos,mut_ext = mutate(ext,pos,voxel_pos,voxel_extents)
#     # pos, ext = slot_arrays_mod(dims/2, dims/2 + np.array([.01, .01, 0]))
#     rgb = [(.75, .25, .25)] * n_parts
#     boxes = list(zip(map(tuple, pos), ext, rgb))
#     boxes2 = list(zip(map(tuple, mut_pos), mut_ext, rgb))
#     obj = add_box_compound(boxes)
#     obj_mutant = add_box_compound(boxes2)
#     t_pos = table_position()
#     t_ext = table_half_extents()
#     s_pos = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - maxz)
#     s_pos2 = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - maxz) + np.random.randn(
#         3) * np.array([0.5, 0, 0])
#     pb.resetBasePositionAndOrientation(obj, s_pos, (0.0, 0.0, 0.0, 1))
#     pb.resetBasePositionAndOrientation(obj_mutant, s_pos2, (0.0, 0.0, 0.0, 1))
#
#     if loop_counter==0:
#         angles = env.angle_dict(env.get_position())
#     angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
#     env.set_position(env.angle_array(angles))
#
#     #Dataset.AddObjtoDict(loop_counter,pos,ext,n_parts)
#     #test = xmltodict.unparse(Dataset.Dict_positions,expand_iter="coord")
#     #test_dict = xmltodict.parse(test)
#
#     #print(tostring(test))
#
#     pos = pb.getBasePositionAndOrientation(obj)
#     quat = pb.getMatrixFromQuaternion(pb.getBasePositionAndOrientation(obj)[1])
#     for i in range(n_parts):
#         LinkInfo = pb.getLinkState(obj,i)
#         Link_pos = list(LinkInfo[0])
#         link_orn = list(LinkInfo[1])
#         quat = pb.getMatrixFromQuaternion(pb.getLinkState(obj,i)[1])
#
#
#
#         #Move above target
#         target_pos = copy.deepcopy(Link_pos)
#         target_pos[2] = target_pos[2] +0.1
#         tarargs = get_tip_targets(target_pos,quat,0.014)
#         i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
#         env.goto_position(i_k, 1)
#         print("Moving-1")
#
#         #move to target
#
#         target_pos = copy.deepcopy(Link_pos)
#         tarargs = get_tip_targets(target_pos, quat, 0.014)
#         i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
#         env.goto_position(i_k, 1)
#         print("Moving-2")
#
#         #Close gripper
#         tarargs = get_tip_targets(target_pos, quat, 0.002)
#         i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
#         env.goto_position(i_k, 1)
#         print("Moving -3")
#
#
#         # Lift
#
#         #Check
#         #if(pb.getBasePositionAndOrientation(obj)[0][2]> 0.1):
#            # print("Object lift Successful")
#
#        # else:
#           #  print("Object lift Unsuccessful")
#     #
#     #
#     pb.removeBody(obj)
# Postoxmlconvert = dict_to_xml('Position',Dataset.Dict_positions)
# Exttoxmlconvert = dict_to_xml('Extent',Dataset.Dict_extents)
# Restoxmlconvert = dict_to_xml('Result',Dataset.Dict_extents)
#
#
#
# import matplotlib.pyplot as plt
#
# plt.scatter(Error_count,Error)
# plt.show()self
#
# input("..")