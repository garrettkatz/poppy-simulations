from dm_control import mujoco
import matplotlib.pyplot as pt

# simplest needs meshes and urdf in same folder
# might be able to use assets kwarg if you preload xml and stl data into python strings
physics = mujoco.Physics.from_xml_path("../urdfs/ergo/meshes/poppy_ergo.dmcontrol.urdf")
pixels = physics.render()

pt.imshow(pixels)
pt.show()

