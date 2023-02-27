if __name__ == "__main__":

    init_range = np.linspace(np.pi/100, np.pi/6, 10)
    shift_swing_range = np.linspace(np.pi/100, np.pi/4, 10)
    push_

    # this launches the simulator
    env = PoppyErgoEnv(pb.POSITION_CONTROL, show=False)

        waypoints = get_waypoints(env,
            # angle from vertical axis to flat leg in initial stance
            init_flat = .02*np.pi,
            # angle for abs_y joint in initial stance
            init_abs_y = np.pi/16,
            # angle from swing leg to vertical axis in shift stance
            shift_swing = .05*np.pi,
            # angle of torso towards support leg in shift stance
            shift_torso = np.pi/5.75,
            # angle from vertical axis to flat leg in push stance
            push_flat = -.02*np.pi,#-.05*np.pi,
            # angle from swing leg to vertical axis in push stance
            push_swing = -.08*np.pi,#-.01*np.pi,
        )

        in_limits, max_error, com_support, clearance, = check_waypoints(env, waypoints)


    pt.rcParams["text.usetex"] = True
    pt.rcParams['font.family'] = 'serif'


