# Poppy Environment

​	Here we provide two Gym-based poppy environments, `PoppyKeepStanding-v0` and `PoppyStandup-v0`.



### How to install the environment

​	The environments are included in `envs` folder and the steps for installing them are as follows.

1. Find the location of the package Gym in your computer. For example, mine is 

   `GYM_PATH =~/anaconda3/envs/rrc_simulation/lib/python3.6/site-packages/gym `

2. Under the folder `GYM_PATH/env/`, register the environment in `__init__.py`.

   For `PoppyKeepStanding-v0`:
   
   `register(
       id = 'PoppyKeepStanding-v0',
       entry_point = 'gym.envs.mujoco.poppy_humanoid_keep_standing:PoppyHumanoidKeepStandingEnv',
       max_episode_steps = 1000,
   )`
   
   For `PoppyStandup-v0`:
   
   `register(
       id = 'PoppyStandup-v0',
       entry_point = 'gym.envs.mujoco.poppy_humanoid_standup:PoppyHumanoidStandupEnv',
       max_episode_steps = 1000,
   )`
   
3. Import the environment class in `GYM_PATH/env/mujoco/__init__.py`.

   For `PoppyKeepStanding-v0`:

   `from gym.envs.mujoco.poppy_humanoid_keep_standing import PoppyHumanoidKeepStandingEnv`

   For `PoppyStandup-v0`:

   `from gym.envs.mujoco.poppy_humanoid_standup import PoppyHumanoidStandupEnv`

4. Put the environment file into the folder `GYM_PATH/envs/mujoco/`.

   For `PoppyKeepStanding-v0`:

   `env/poppy_humanoid_keep_standing/poppy_humanoid_keep_standing.py`

   For `PoppyStandup-v0`: 

   `env/poppy_humanoid_standup/poppy_humanoid_standup.py`

5. Put the asset folder `poppyhumanoid` into `GYM_PATH/envs/mujoco/assets/`.

6. Test the environment by running `python run_env.py`.



​	If the configuration is successful, then you can see

`PoppyKeepStanding-v0`

![PoppyHumanoid_keep_standing_test](./envs/poppy_humanoid_keep_standing/PoppyHumanoid_keep_standing_test.gif)

`PoppyStandup-v0`

![PoppyHumanoid_standup_test](./envs/poppy_humanoid_standup/PoppyHumanoid_standup_test.gif)



### Notes

​	If you meet any problems, refer to the notes under `notes/` and search for possible solutions.