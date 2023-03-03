import matplotlib.pyplot as pt
import itertools as it
import sys, os
import pybullet as pb
import numpy as np
import pickle as pk

# custom modules paths, should eventually be packaged
sys.path.append(os.path.join('..', '..', 'envs'))

# common
from ergo import PoppyErgoEnv

from phase_waypoints import phase_waypoint_figure, get_waypoints, check_waypoints

if __name__ == "__main__":

    pt.rcParams["text.usetex"] = True
    pt.rcParams['font.family'] = 'serif'

    do_shift_filter = False
    do_push_filter = False
    do_both_filter = False
    show_filter = True

    grid = 8

    # angle from vertical axis to flat leg in initial stance
    init_flat_range = np.linspace(0.01*np.pi, 0.08*np.pi, grid) # .02*np.pi
    # angle for abs_y joint in initial stance
    init_abs_y = np.pi/16
    # angle from swing leg to vertical axis in shift stance
    shift_swing_range = np.linspace(0.03*np.pi, 0.10*np.pi, grid)#.05*np.pi
    # angle of torso towards support leg in shift stance
    shift_torso = np.pi/5
    # angle from vertical axis to flat leg in push stance
    push_flat_range = np.linspace(-.03*np.pi, .04*np.pi, grid)#-.00*np.pi + np.random.normal(0, 0.5 * np.pi/180)
    # angle from swing leg to vertical axis in push stance
    push_swing_range = np.linspace(-.14*np.pi, -.07*np.pi, grid)#-.10*np.pi

    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)

    if do_shift_filter:

        in_limits = np.empty((len(init_flat_range), len(shift_swing_range)), dtype=bool)
        max_error = np.empty((len(init_flat_range), len(shift_swing_range)), dtype=float)
        com_support = np.empty((len(init_flat_range), len(shift_swing_range)), dtype=bool)

        for k, (init_flat, shift_swing) in enumerate(it.product(init_flat_range, shift_swing_range)):
            print(f"{k} of {len(init_flat_range)*len(shift_swing_range)}...")
    
            params = (init_flat, init_abs_y, shift_swing, shift_torso, push_flat_range[0], push_swing_range[0])
            waypoints = get_waypoints(env, *params, only_shift=True)
            in_limits.flat[k], max_error.flat[k], com_support.flat[k], _ = check_waypoints(env, waypoints)

        with open('filter_shift_{grid}.pkl','wb') as f: pk.dump((in_limits, max_error, com_support), f)

    if do_push_filter:

        in_limits = np.empty((len(push_flat_range), len(push_swing_range)), dtype=bool)
        max_error = np.empty((len(push_flat_range), len(push_swing_range)), dtype=float)
        com_support = np.empty((len(push_flat_range), len(push_swing_range)), dtype=bool)
        clearance = np.empty((len(push_flat_range), len(push_swing_range)), dtype=bool)

        for k, (push_flat, push_swing) in enumerate(it.product(push_flat_range, push_swing_range)):
            print(f"{k} of {len(push_flat_range)*len(push_swing_range)}...")
    
            params = (.02*np.pi, init_abs_y, .05*np.pi, shift_torso, push_flat, push_swing)
            waypoints = get_waypoints(env, *params)
            in_limits.flat[k], max_error.flat[k], com_support.flat[k], clearance.flat[k] = check_waypoints(env, waypoints)

        with open('filter_push_{grid}.pkl','wb') as f: pk.dump((in_limits, max_error, com_support, clearance), f)

    if do_both_filter:

        in_limits = np.empty((len(init_flat_range), len(push_flat_range), len(push_swing_range)), dtype=bool)
        max_error = np.empty((len(init_flat_range), len(push_flat_range), len(push_swing_range)), dtype=float)
        com_support = np.empty((len(init_flat_range), len(push_flat_range), len(push_swing_range)), dtype=bool)
        clearance = np.empty((len(init_flat_range), len(push_flat_range), len(push_swing_range)), dtype=bool)

        with open('filter_shift_{grid}.pkl','rb') as f: (_, _, shift_com_support) = pk.load(f)

        for i, init_flat in enumerate(init_flat_range):
            # shift_swing = shift_swing_range[shift_com_support[i].argmax()]
            shift_swing = 2.5*init_flat
            for k, (push_flat, push_swing) in enumerate(it.product(push_flat_range, push_swing_range)):
                print(f"{i},{k} of {len(init_flat_range)},{len(push_flat_range)*len(push_swing_range)}...")
        
                params = (init_flat, init_abs_y, shift_swing, shift_torso, push_flat, push_swing)
                waypoints = get_waypoints(env, *params)
                in_limits[i].flat[k], max_error[i].flat[k], com_support[i].flat[k], clearance[i].flat[k] = check_waypoints(env, waypoints)

        with open('filter_both_{grid}.pkl','wb') as f: pk.dump((in_limits, max_error, com_support, clearance), f)

    if show_filter:

        with open('filter_shift_{grid}.pkl','rb') as f: (in_limits, max_error, com_support) = pk.load(f)

        fig, ax = pt.subplots(1, 3, figsize=(4,1.6), constrained_layout=True)
        ax[0].imshow(in_limits, vmin=0, vmax=1, cmap='gray')
        ax[0].set_title('In Limits')
        ax[1].imshow(com_support, vmin=0, vmax=1, cmap='gray')
        ax[1].set_title('CoM Support')
        im = ax[2].imshow(max_error, vmin=0, cmap='gray')
        ax[2].set_title('IK Error')
        fig.colorbar(im, ax=ax[2])
        for a in ax:
            a.set_xticks([0.5, len(shift_swing_range)-.5], [f"{shift_swing_range[i]:.2f}" for i in [0, -1]])
            a.set_yticks([], [])
        ax[0].set_yticks([0.5, len(init_flat_range)-.5], [f"{init_flat_range[i]:.2f}" for i in [0, -1]])
        # ax[1].set_xlabel("$\\theta^{(\\mathbf{S})}_s$", fontsize=14)
        fig.supxlabel("$\\theta^{(\\mathbf{S})}_s$", fontsize=14)
        fig.supylabel("$\\theta^{(\\mathbf{I})}_f$", rotation=0, fontsize=14)
        pt.savefig(f'shift_filter_{grid}.eps')
        pt.show()

        with open('filter_push_{grid}.pkl','rb') as f: (in_limits, max_error, com_support, clearance) = pk.load(f)

        fig, ax = pt.subplots(1, 4, figsize=(6,1.6), constrained_layout=True)
        ax[0].imshow(in_limits, vmin=0, vmax=1, cmap='gray')
        ax[0].set_title('In Limits')
        ax[1].imshow(com_support, vmin=0, vmax=1, cmap='gray')
        ax[1].set_title('CoM Support')
        ax[2].imshow(clearance, vmin=0, vmax=1, cmap='gray')
        ax[2].set_title('Clearance')
        im = ax[3].imshow(max_error, vmin=0, cmap='gray')
        ax[3].set_title('IK Error')
        fig.colorbar(im, ax=ax[3])
        for a in ax:
            a.set_xticks([0.5, len(push_swing_range)-.5], [f"{push_swing_range[i]:.2f}" for i in [0, -1]])
            a.set_yticks([], [])
        ax[0].set_yticks([0.5, len(push_flat_range)-.5], [f"{push_flat_range[i]:.2f}" for i in [0, -1]])
        # ax[1].set_xlabel("$\\theta^{(\\mathbf{P})}_s$", fontsize=14)
        fig.supxlabel("$\\theta^{(\\mathbf{P})}_s$", fontsize=14)
        fig.supylabel("$\\theta^{(\\mathbf{P})}_f$", rotation=0, fontsize=14)
        pt.savefig(f'push_filter_{grid}.eps')
        pt.show()

        with open('filter_both_{grid}.pkl','rb') as f: (in_limits, max_error, com_support, clearance) = pk.load(f)

        fig, ax = pt.subplots(2, len(init_flat_range), figsize=(13,3.5), constrained_layout=True)
        for i, init_flat in enumerate(init_flat_range):
            # feas = in_limits[i] + com_support[i] + clearance[i]
            # ax[0,i].imshow(feas, vmin=0, vmax=3, cmap='gray')
            feas = in_limits[i] & com_support[i] & clearance[i]
            ax[0,i].imshow(feas, vmin=0, vmax=1, cmap='gray')
            im_i = ax[1,i].imshow(max_error[i], vmin=0, vmax=max_error.max(), cmap='gray')
            # im_i = ax[1,i].imshow(np.log(max_error[i]), vmin=np.log(max_error).min(), vmax=np.log(max_error).max(), cmap='gray')
            if max_error[i].max() == max_error.max(): im = im_i
            if i == 0:
                ax[0,i].set_yticks([0.5, len(push_flat_range)-.5], [f"{push_flat_range[k]:.2f}" for k in [0, -1]])
                ax[1,i].set_yticks([0.5, len(push_flat_range)-.5], [f"{push_flat_range[k]:.2f}" for k in [0, -1]])
            else:
                ax[0,i].set_yticks([], [])
                ax[1,i].set_yticks([], [])
            ax[0,i].set_title(f"{init_flat:.2f}")
            ax[0,i].set_xticks([], [])
            ax[1,i].set_xticks([0.5, len(push_swing_range)-.5], [f"{push_swing_range[k]:.2f}" for k in [0, -1]])
        ax[0,0].set_ylabel('Feasibility')
        ax[1,0].set_ylabel('IK Error')
        fig.suptitle("$\\theta^{(\mathbf{I})}_f$")
        fig.supxlabel("$\\theta^{(\mathbf{P})}_s$")
        fig.supylabel("$\\theta^{(\mathbf{P})}_f$", rotation=0)
        fig.colorbar(im, ax=ax[:,-1])
        pt.savefig(f'both_filter_{grid}.eps')
        pt.show()


        # params = (init_flat_range[-1], init_abs_y, shift_swing_range[-1], shift_torso, push_flat_range[0], push_swing_range[0])
        # waypoints = get_waypoints(env, *params)
        # phase_waypoint_figure(env, waypoints)


