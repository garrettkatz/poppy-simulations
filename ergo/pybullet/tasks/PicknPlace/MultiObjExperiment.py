import MultObjPick
import BaselineLearner

import numpy as np



#Learner = BaselineLearner()

Experiment_env = MultObjPick.experiment()
Experiment_env.CreateScene()

dims = np.array([.01, .01, .01])
n_parts = 6
rgb = [(.75, .25, .25)] * n_parts
Experiment_obj = MultObjPick.Obj(dims,n_parts,rgb)
Experiment_obj.GenerateObject(dims,n_parts,[0,0,0])
Experiment_env.Spawn_Object(Experiment_obj)
Experiment_env.MoveToPos(Experiment_obj.basePosition,0.005)
print("")