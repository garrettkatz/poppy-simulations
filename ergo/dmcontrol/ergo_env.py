import os
from dm_control import mjcf
from dm_env.specs import BoundedArray
from dm_env import Environment, StepType, TimeStep

class ErgoDomain:
    def __init__(self, control_period):
        self.control_period = control_period # number of physics steps per action
        self.physics = None
        # physics.model.nq

    def reset(self, kp = None):

        # load ergo model
        # first do `mujoco/bin/path/compile poppy_ergo.dmcontrol.urdf poppy_ergo.dmcontrol.xml`
        # mjcf_path = os.path.join("..","urdfs","ergo","poppy_ergo.dmcontrol.xml")
        mjcf_path = os.path.join("..","urdfs","ergo","poppy_ergo.dmcontrol.mod.xml")
        self.arena = mjcf.from_path(mjcf_path)

        # Domain tweak
        if kp is not None:
            for p in range(len(self.arena.actuator.position)):
                self.arena.actuator.position[p].kp = kp[p]

        # add floor from https://arxiv.org/pdf/2006.12983.pdf
        checker = self.arena.asset.add(
            'texture', type='2d', builtin='checker', width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = self.arena.asset.add('material', name='grid', texture=checker, texrepeat=[5,5], reflectance=.2)
        self.arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        # extract joint information for action spec
        actuators = self.arena.find_all("actuator")
        joints = [a.joint for a in actuators]
        ranges = [j.range for j in joints]
        self._action_spec = BoundedArray(
            shape=(len(joints),),
            dtype=float,
            minimum=[r[0] for r in ranges],
            maximum=[r[1] for r in ranges],
        )

        # set up physics and return initial observation
        self.physics = mjcf.Physics.from_mjcf_model(self.arena)

    def action_spec(self):
        return self._action_spec

    def step(self, action=None):
        for _ in range(self.control_period):
            self.physics.step()

class ErgoEnv(Environment):
    def __init__(self, control_period):
        self.domain = ErgoDomain(control_period)

    def action_spec(self):
        return self.domain.action_spec()

    def observation_spec(self):
        # same as action spec for position joint control
        return self.domain.action_spec()

    def get_observation(self):
        return self.domain.physics.data.qpos.copy()

    def reset(self, kp = None):
        self.domain.reset(kp)
        self.physics = self.domain.physics
        reward, discount = 0, None
        return TimeStep(StepType.FIRST, reward, discount, self.get_observation())

    def step(self, action=None):
        self.domain.step(action)
        reward, discount = 0, 1
        return TimeStep(StepType.MID, reward, discount, self.get_observation())


if __name__ == "__main__":

    import numpy as np

    env = ErgoEnv(control_period = 10)

    kp = 30*np.ones(31)
    # kp = 20 + 30000*np.ones(31)
    ts = env.reset(kp)

    # model names, but changes don't seem to affect phys
    # env.physics.named.model.dof_armature
    # env.physics.named.model.dof_damping
    # env.physics.named.model.actuator_gainprm
    # env.physics.named.model.actuator_biasprm
    # env.physics.named.model.body_inertia
    # print(env.physics.named.model.actuator_gainprm)

    def zero_policy(ts):
        # print(ts.observation)
        np.zeros(env.action_spec().shape)

    # using viewer with conformant env
    from dm_control import viewer
    viewer.launch(env, zero_policy)

    # # raw matplotlib viz without conformant env
    # import matplotlib.pyplot as pt
    # pt.ion()
    # pt.show()
    # for t in range(50):
    #     obs = env.step()
    #     pixels = env.physics.render()
    #     pt.cla()
    #     pt.imshow(pixels)
    #     pt.pause(.01)
    #     print(t)
    #     # input(t)
    # pt.close()

