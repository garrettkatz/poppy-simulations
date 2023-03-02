# Walking Gait Synthesis and Execution

Getting Poppy to autonomously take a step forward involves two stages:

- Synthesis of the joint trajectory

- Execution of the joint trajectory

Synthesis is done using the code in this repository and execution is done using code in an adjoining repository as described [here](https://github.com/garrettkatz/poppy-muffin/blob/master/scripts/walk.md).  You can synthesize trajectories as follows:

1. Run the command

        $ python phase_trajectories.py

    This will create a trajectory using hand-tuned parameters, display some figures to visualize it, and write out the trajectory data to the file `pypot_traj1.pkl`.  It can also visualize the trajectory as a PyBullet simulation.  To control whether figures and simulation are displayed, you can modify the `show_traj` and `run_traj` flags at the top of the `__main__` block.  Right below that you can also modify the hand-tuned trajectory parameters in the call to `get_waypoints(...)`.

2. Run the command

        $ python experiment_trajectories.py

    This will create multiple trajectories using random perturbations to the hand-tuned parameters, and write each one to a corresponding file `exp_normal/pypot_sample_trajectory_<sample>.pkl`.  You can modify the number of sampled perturbations near the top of the `__main__` block.

