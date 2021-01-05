# PoppyHumanoid Environment

Steps for creating a PoppyHumanoid environment:

1. Find the location of the package Gym in your computer. For example, mine is 

   `GYM_PATH =~/anaconda3/envs/rrc_simulation/lib/python3.6/site-packages/gym `

2. Under the folder `GYM_PATH/env/`, register the environment in `__init__.py` by adding the following commands.

   `register(
       id = 'PoppyHumanoid-v0',
       entry_point = 'gym.envs.mujoco.poppy_humanoid:PoppyHumanoidEnv',
       max_episode_steps = 1000,)`
       
3. Import the environment class in `GYM_PATH/env/mujoco/__init__.py` by
   `from gym.envs.mujoco.poppy_humanoid import PoppyHumanoidEnv`

4. Put `poppy_humanoid.py` into `GYM_PATH/envs/mujoco/`.

5. Put the folder `poppyhumanoid` into `GYM_PATH/envs/mujoco/assets/`.

6. Test the environment by running `python run_poppyhumanoid_env.py`.



If the configuration is successful, then you can see

![](./PoppyHumanoid_stand_env_test.gif)
