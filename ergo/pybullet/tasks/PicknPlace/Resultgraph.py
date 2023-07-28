import pickle
import glob, os
import re

import numpy as np
import matplotlib as plt


WorkingDir = os.getcwd()
os.chdir(WorkingDir)

for file in os.listdir(WorkingDir):
    if file.endswith(".pickle"):
        match = re.search(r'\d+',file)
        if match:
            data = pickle.load(file)
            if match:
                x= np.empty(len(data))
                x.fill(match.group())
                plt.scatter(x,data)
            else:
                x = np.empty(len(data))
                x.fill(0)
                plt.scatter(x,data)

plt.show()