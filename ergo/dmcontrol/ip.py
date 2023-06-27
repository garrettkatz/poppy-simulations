import numpy as np
import matplotlib.pyplot as pt
import matplotlib.animation as animation

from dm_control import mjcf

# floor
arena = mjcf.RootElement()
chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                            height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
grid = arena.asset.add('material', name='grid', texture=chequered,
                       texrepeat=[5, 5], reflectance=.2)
arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)
for x in [-2, 2]:
  arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])

# model defaults
model = mjcf.RootElement()
model.compiler.angle = 'radian'  # Use radians.

model.default.joint.damping = 0
model.default.joint.type = 'hinge'
model.default.geom.type = 'capsule'
model.default.geom.rgba = (.25, .5, .75, 1)

length = 1
sz = length/8

# base joint
base = model.worldbody.add('body', name='base')
fulcrum = base.add('joint', axis=[0,1,0], name='fulcrum')
base.add('geom', type='sphere', size=[sz])

# link
pendulum = base.add('body', name='rod')
rod = pendulum.add('geom', fromto=[0,0,0, 0,0,length], size=[sz])

# control
model.actuator.add('position', joint=fulcrum, kp=0)
actuators = model.find_all('actuator')

# attach to floor
spawn_site = arena.worldbody.add('site', pos=(0,0,sz), group=3)
spawn_site.attach(model)

# initial angle
physics = mjcf.Physics.from_mjcf_model(arena)

with physics.reset_context():
    physics.data.qpos[0] = np.pi/4

# run simulation
print('sim...')
video = []
for t in range(300):

    print(t, physics.data.qpos)
    # print(actuators)
    # print(physics.named.data.xfrc_applied['unnamed_model/rod'])

    for s in range(5):
        if t < 5:
            physics.named.data.xfrc_applied['unnamed_model/rod'][2] = 10000
        else:
            physics.named.data.xfrc_applied['unnamed_model/rod'][2] = 0
        # physics.bind(actuators).ctrl = [-.3]
        physics.step()

    arr = physics.render()
    video.append(arr.copy())

# render video
fig = pt.gcf()
ax = pt.gca()
video = [[ax.imshow(arr, animated=True)] for arr in video]
ani = animation.ArtistAnimation(fig, video, interval=50, blit=True, repeat_delay=1000, repeat=False)
pt.show()

# pt.ion()
# pt.show()
# for arr in video:
#     pt.cla()
#     pt.imshow(arr)
#     pt.pause(0.01)


