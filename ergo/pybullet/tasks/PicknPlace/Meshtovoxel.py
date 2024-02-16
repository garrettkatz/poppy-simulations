#import stl_reader
import stltovoxel
import numpy as np
import copy
import os
import sys
cwr = os.getcwd()
stlpath = cwr+"/Stanford_Bunny_sample.stl"
stlpath2 = cwr+"/Eiffel_tower_sample.STL"
outputpath = cwr+"/stlresultbunny.xyz"
npyout = cwr+"/npyout.npy"
res =50

stltovoxel.convert_file(stlpath2, npyout,resolution=100,voxel_size=10)
#stltovoxel.convert.export_npy(stlpath,outputpath,scale=10,shift=1)
res = np.load(npyout)


print(outputpath)