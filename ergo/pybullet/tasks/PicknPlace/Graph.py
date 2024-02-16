import pickle
import glob, os
import re
import numpy as np
import matplotlib.pyplot as plt
WorkingDir = os.getcwd() #+ "/Feb12Res_cond_iou_co13"
os.chdir(WorkingDir)
linegraph_x = []
linegraph_y = []
# Obj.pickle and gen1 res before that
#violin plot






f = open(f'mutant_Gen0_numgrip_results.pickle','rb')
data = pickle.load(f)
f.close()
flatlist = []

npdata = np.array(data)
npdata2 = npdata[npdata != 0]
#npdata = npdata.squeeze(axis=0)
x = np.empty(len(npdata2))
x.fill(0)
plt.scatter(x,npdata2)
linegraph_x.append(0)
linegraph_y.append(np.average(npdata2))

#Violinplotlist = []
#f = open(f'Gen1_results.pickle','rb')
#data = pickle.load(f)
#f.close()
#flatlist = []

npdata = data
#npdata = npdata.squeeze(axis=0)
x = np.empty(len(npdata))
x.fill(0)
plt.scatter(x,npdata)
linegraph_x.append(1)
linegraph_y.append(np.average(npdata))
#linegraph_z.append(np.average(npdata))
#Violinplotlist.append(npdata)

for i in range(1,29):
    f = open(f'mutantGen_{i}_results.pickle','rb')
    data = pickle.load(f)
    f.close()
    flatlist = []
    #for sublist in data:
       #for element in sublist:
           # for e in element:
               # flatlist.append(e)
    npdata = data
    #npdata = npdata.squeeze(axis=0)
    x = np.empty(len(npdata))
    x.fill(i+1)
    plt.scatter(x,npdata)

    #plt.show()

    linegraph_x.append(i+1)
    linegraph_y.append(np.average(npdata))
    #linegraph_z.append(np.average(npdata))
plt.plot(linegraph_x,linegraph_y,c='r')
#plt.plot(i,linegraph_z,c='g')
#

plt.show()
print("")