# Poppy Ergo Jr URDF

URDF is the [Unified Robot Description Format](http://wiki.ros.org/urdf). It describes the kinematic chain of your robot.

This package generates the urdf file at runtime by invoking [xacro](http://wiki.ros.org/xacro).
The generated URDF must match the the way you've assembled your Ergo Jr, i.e. with the gripper or with the lamp effector.
Currently the pen holder effector is not supported: if you can, please make a pull request!

## Generate the URDF by choosing your effector (preferred)

Invoke xacro by passing the desired effector as an argument and redirect the output to an URDF file:
```
roscd poppy_ergo_jr_description/urdf

xacro poppy_ergo_jr.urdf.xacro gripper:=true   >poppy_ergo_jr.urdf
xacro poppy_ergo_jr.urdf.xacro lamp:=true      >poppy_ergo_jr.urdf
```

## Use a static URDF with the gripper effector (deprecated)

The static file [poppy_ergo_jr.urdf](./poppy_ergo_jr.urdf) is the output of xacro generated for the gripper effector.

## Compatibility

This URDF has been tested in ROS Melodic and ROS Noetic. Older versions may need the `--inline` parameter for xacro.
