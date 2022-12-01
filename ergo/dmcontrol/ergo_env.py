import os
from dm_control import mjcf

class ErgoEnv:
    def __init__(self, control_period):
        self.control_period = control_period # number of physics steps per action
        self.physics = None
        # physics.model.nq

    def get_observation(self):
        return self.physics.data.qpos.copy()

    def reset(self):

        # load ergo model
        # first do `mujoco/bin/path/compile poppy_ergo.dmcontrol.urdf poppy_ergo.dmcontrol.xml`
        # mjcf_path = os.path.join("..","urdfs","ergo","poppy_ergo.dmcontrol.xml")
        mjcf_path = os.path.join("..","urdfs","ergo","poppy_ergo.dmcontrol.mod.xml")
        arena = mjcf.from_path(mjcf_path)

        # add floor from https://arxiv.org/pdf/2006.12983.pdf
        checker = arena.asset.add(
            'texture', type='2d', builtin='checker', width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = arena.asset.add('material', name='grid', texture=checker, texrepeat=[5,5], reflectance=.2)
        arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        # set up physics and return initial observation
        self.physics = mjcf.Physics.from_mjcf_model(arena)
        return self.get_observation()

    def step(self, action=None):
        for _ in range(self.control_period):
            self.physics.step()
        return self.get_observation()


if __name__ == "__main__":

    import matplotlib.pyplot as pt
    
    env = ErgoEnv(control_period = 10)
    obs = env.reset()

    # maybe through pymjcf, and/or with urdf extensions (also for meshdir):
    # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=urdf#urdf-extensions

    pt.ion()
    pt.show()
    for t in range(50):
        obs = env.step()
        pixels = env.physics.render()

        pt.cla()
        pt.imshow(pixels)
        pt.pause(.01)

        print(t)
        # input(t)

    pt.close()

