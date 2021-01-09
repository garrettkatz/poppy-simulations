# Problems in customizing an environment

## About Gym

**Q**: How to customize an environment?

**A**: The steps for customizing an environment are

1. Find the location `GYM_PATH` of the package Gym in your computer.
2. Register the environment.
3. Prepare for the code and asset of the customized environment and put them under `GYM_PATH/envs`.
4. Tweak the parameters of your environment.



**Q**: Any libraries for implemented reinforcement learning algorithms?

**A**: [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/). 



## About MuJoCo

**Q**: How to create your own MuJoCo XML model?

**A**: The steps for customizing your MuJoCo model are

1. Have a URDF model file. The included mesh files should be STL format and use absolute path.
2. Check the grammar of your URDF file: `check_urdf /path/model.urdf`.
3. Go to the directory where MuJoCo is installed and compile the URDF file: `./compile /path/model.urdf /path/model.xml`. 
4. Test whether the compilation is successful: `./simulate /path/model.xml`.
5. However, the generated model has no physical properties. Add more attributes by modifying the XML file (details in [MuJoCo XML Reference](http://www.mujoco.org/book/XMLreference.html)).



**Q**: How to add actuators (a physical property) to a XML model?

**A**: Add an `actuator` label after `worldbody`. The format is as follows, where `joint_name` is the name of joint in the model.

```xml
<actuator>
    <motor name="xxx" joint="joint_name"/>
</actuator>
```



**Q**: What is the difference between the functions `load_model_from_path()` and `load_model_from_xml()` in MuJoCo-py?

**A**: The first function loads the model from a given file, while the second function loads the model from a string of XML format. I mistook these two functions and used `load_model_from_xml()` to load from a XML file (of course failed).



**Q**: Why the model doesn't fall to the ground even if claiming the gravity?

**A**: Add a `freejoint` label inside `body` label of the model as follows. However, if the other labels at the same level are `body`, then `mass and inertia must be positive` error will be reported. Now I don't figure out how to solve this problem, but adding a `geom` label can avoid it.

```xml
<worldbody>
    <body ...>
        <freejoint/>
        <geom>
        <body>
        ...
    </body>
</worldbody>
```