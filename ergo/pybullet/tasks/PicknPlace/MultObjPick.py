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
                voxel_positions.append(new_base2)
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
        self.isMutant = False
        #or self.mutants = list of mutants
        self.ParentId = -1

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
        self.positions =positions
        self.extents = extents
        self.maxz = maxz
        return self


    def MutateObject(self):
        new_mutant_obj_pos = self.positions.copy()
        print(new_mutant_obj_pos[len(new_mutant_obj_pos) - 1])
        print(np.random.randint(0, len(voxel_pos) - 1))
        print(self.PositionsAvailable[np.random.randint(0, len(voxel_pos) - 1)])
        self.PositionsAvailable,_ = self.GetOpenPosition()

        new_mutant_obj_pos[len(new_mutant_obj_pos) - 1] = Obj.PositionsAvailable[
            np.random.randint(0, len(voxel_pos) - 1)]  # worst location on fitness function
        newobj = Obj(new_mutant_obj_pos,self.extents,self.NoOfParts,self.rgb)
        newobj.isMutant= True
        newobj.ParentId = 1
        return newobj


    def SpawnObject(self,env,dim,noofparts,basepos):
        self.GenerateObject(dim,noofparts,basepos)
        Boxes = list(zip(map(tuple, self.positions), self.extents, self.rgb))
        ObjInfo = add_box_compound(Boxes)
        return ObjInfo

    def GetOpenPosition(self):
        positions = self.positions
        extents = self.extents
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
                    voxel_positions.append(new_base2)
        self.PositionsAvailable = voxel_positions.copy()
        return self



dims = np.array([.01, .01, .01])
n_parts = 6
rgb = [(.75, .25, .25)] * n_parts
obj = Obj(dims,n_parts,rgb)
obj = obj.GenerateObject(dims,n_parts,[0,0,0])
obj = obj.SpawnObject(dims,n_parts,[0,0,0])


if __name__ == "__main__":
    sys.path.append(os.path.join('..', '..', 'envs'))
    from ergo import PoppyErgoEnv
    import motor_control2 as mc
    sys.path.append(os.path.join('..', '..', 'objects'))
    from tabletop import add_table, add_cube,add_obj,add_box_compound,table_position,table_half_extents

    # this launches the simulator
    # fix the base to avoid balance issues : Removed in Update
    env = PoppyErgoEnv(pb.POSITION_CONTROL, use_fixed_base=True)
    # this adds the table
    add_table()

    # add a few cubes at random positions on the table
Error = []
Error_count=[]
urdfList = []
fpath = "D:/G/Poppy_data/**/*.urdf"
files = glob.glob(fpath,recursive=True)
test = np.random.randn()


class experiment:
    def __init__(self,env):
        self.env = env

    def CreateScene(self):
        return 0




Dataset = Data_Dict()

for loop_counter in range(100):
    n_parts = 6
    pos, ext, maxz = Generative_Shapes(dim, n_parts, [0, 0, 0])
    voxel_extents,voxel_pos = ObjectOpenPositions(pos,ext)
    dims = np.array([.01, .01, .01])
    mut_pos,mut_ext = mutate(ext,pos,voxel_pos,voxel_extents)
    # pos, ext = slot_arrays_mod(dims/2, dims/2 + np.array([.01, .01, 0]))
    rgb = [(.75, .25, .25)] * n_parts
    boxes = list(zip(map(tuple, pos), ext, rgb))
    boxes2 = list(zip(map(tuple, mut_pos), mut_ext, rgb))
    obj = add_box_compound(boxes)
    obj_mutant = add_box_compound(boxes2)
    t_pos = table_position()
    t_ext = table_half_extents()
    s_pos = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - maxz)
    s_pos2 = (t_pos[0], t_pos[1] + t_ext[1] / 2, t_pos[2] + t_ext[2] + dims[2] / 2 - maxz) + np.random.randn(
        3) * np.array([0.5, 0, 0])
    pb.resetBasePositionAndOrientation(obj, s_pos, (0.0, 0.0, 0.0, 1))
    pb.resetBasePositionAndOrientation(obj_mutant, s_pos2, (0.0, 0.0, 0.0, 1))

    if loop_counter==0:
        angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
    env.set_position(env.angle_array(angles))

    #Dataset.AddObjtoDict(loop_counter,pos,ext,n_parts)
    #test = xmltodict.unparse(Dataset.Dict_positions,expand_iter="coord")
    #test_dict = xmltodict.parse(test)

    #print(tostring(test))

    pos = pb.getBasePositionAndOrientation(obj)
    quat = pb.getMatrixFromQuaternion(pb.getBasePositionAndOrientation(obj)[1])
    for i in range(n_parts):
        LinkInfo = pb.getLinkState(obj,i)
        Link_pos = list(LinkInfo[0])
        link_orn = list(LinkInfo[1])
        quat = pb.getMatrixFromQuaternion(pb.getLinkState(obj,i)[1])



        #Move above target
        target_pos = copy.deepcopy(Link_pos)
        target_pos[2] = target_pos[2] +0.1
        tarargs = get_tip_targets(target_pos,quat,0.014)
        i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
        env.goto_position(i_k, 1)
        print("Moving-1")

        #move to target

        target_pos = copy.deepcopy(Link_pos)
        tarargs = get_tip_targets(target_pos, quat, 0.014)
        i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
        env.goto_position(i_k, 1)
        print("Moving-2")

        #Close gripper
        tarargs = get_tip_targets(target_pos, quat, 0.002)
        i_k = mc.balanced_reach_ik(env, tarargs, arm="right")
        env.goto_position(i_k, 1)
        print("Moving -3")


        # Lift

        #Check
        #if(pb.getBasePositionAndOrientation(obj)[0][2]> 0.1):
           # print("Object lift Successful")

       # else:
          #  print("Object lift Unsuccessful")
    #
    #
    pb.removeBody(obj)
Postoxmlconvert = dict_to_xml('Position',Dataset.Dict_positions)
Exttoxmlconvert = dict_to_xml('Extent',Dataset.Dict_extents)
Restoxmlconvert = dict_to_xml('Result',Dataset.Dict_extents)



import matplotlib.pyplot as plt

plt.scatter(Error_count,Error)
plt.show()

input("..")