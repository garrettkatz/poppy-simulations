import os
from dm_control import mujoco

class ErgoEnv:
    def __init__(self, control_period):
        self.control_period = control_period # number of physics steps per action
        self.physics = None
        # physics.model.nq

    def get_observation(self):
        return self.physics.data.qpos.copy()

    def reset(self):
        urdf_path = os.path.join("..","urdfs","ergo","meshes","poppy_ergo.dmcontrol.urdf")
        self.physics = mujoco.Physics.from_xml_path(urdf_path)
        return self.get_observation()

    def step(self, action=None):
        for _ in range(self.control_period):
            self.physics.step()
        return self.get_observation()


if __name__ == "__main__":

    import matplotlib.pyplot as pt
    
    env = ErgoEnv(control_period = 10)
    obs = env.reset()

    # appears like fixed base joint, need to modify model
    # maybe through pymjcf, and/or with urdf extensions (also for meshdir):
    # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=urdf#urdf-extensions

    pt.ion()
    pt.show()
    for t in range(100):
        obs = env.step()
        pixels = env.physics.render()

        pt.cla()
        pt.imshow(pixels)
        pt.pause(.01)

        print(t)

    pt.close()

