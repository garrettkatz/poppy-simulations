import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from mpl_toolkits.mplot3d import Axes3D

class ZMPWalkPatternGenerator(object):
    def __init__(self, CoM_height = 0.3, foot_height = 0.1, shift_x = 0.1, shift_y = 0.04, T_sup = 0.5, g = 9.81, dT = 5e-3, q = 1, r = 1e-6, preview_steps = 320):
        '''
            Parameters
        '''        
        self.zc = CoM_height
        self.fh = foot_height
        self.sx = shift_x
        self.sy = shift_y
        self.dT = dT
        self.g = g
        self.T_sup = T_sup
        self.sup_steps = int(self.T_sup / self.dT)

        self.N = preview_steps
        self.q = q
        self.r = r

        '''
            Create the system
        '''
        self.A = np.array([
            [1, dT, dT**2/2],
            [0, 1, dT],
            [0, 0, 1]
        ])
        self.B = np.array([[dT**3/6, dT**2/2, dT]]).T
        self.C = np.array([[1, 0, -self.zc/g]])

        '''
            Create the preview controller
        '''
        self.K, self.Fs = self.create_controller()

        '''
            State vector
        '''
        self.X = np.array([0, 0, 0]).T
        self.Y = np.array([0, 0, 0]).T

    '''
        Create ZMP patterns
    '''
    def create_ZMP_pattern(self, N_sup):
        '''
            Generate ZMP positions with given parameters.
            The trajectories:
                X-axis:
                             |----
                        |----|
                    ----|
                Y-axis:
                         |--|
                    --|  |  |  |--
                      |--|  |--|
        '''
        patterns = np.zeros([(N_sup + 2) * self.sup_steps, 3])
        lfoot_traj = np.zeros([(N_sup + 2) * self.sup_steps, 3])
        rfoot_traj = np.zeros([(N_sup + 2) * self.sup_steps, 3])

        lfoot_traj[:, 1] = self.sy
        rfoot_traj[:, 1] = -self.sy

        # Move right foot first.
        dx = -self.sx
        dy = -self.sy
        steps = self.sup_steps
        tmp_x = self.sx * np.linspace(0, 1, num = self.sup_steps)
        tmp_z = self.fh * np.sin(np.linspace(0, 1, num = self.sup_steps) * np.pi)
        for n in range(1, N_sup + 1):
            # ZMP
            dx += self.sx
            dy = -dy
            patterns[steps:steps + self.sup_steps, 0] = dx
            patterns[steps:steps + self.sup_steps, 1] = dy

            # Left foot and right foot
            if n % 2 == 1:
                lfoot_traj[steps:steps + self.sup_steps, 0] = dx

                rfoot_traj[steps:steps + self.sup_steps, 0] = dx + tmp_x if n == 1 else dx - self.sx + 2 * tmp_x
                rfoot_traj[steps:steps + self.sup_steps, 2] = tmp_z
            else:
                lfoot_traj[steps:steps + self.sup_steps, 0] = dx - self.sx + tmp_x if n == N_sup else dx - self.sx + 2 * tmp_x
                lfoot_traj[steps:steps + self.sup_steps, 2] = tmp_z

                rfoot_traj[steps:steps + self.sup_steps, 0] = dx

            steps += self.sup_steps

        patterns[-self.sup_steps:, 0] = dx

        return patterns, lfoot_traj, rfoot_traj
        

    def create_controller(self):
        R = self.r * np.eye(1)
        Q = self.q * self.C.T @ self.C
        P = solve_discrete_are(self.A, self.B, Q, R)

        tmp = np.linalg.inv(R + self.B.T @ P @ self.B) @ self.B.T
        K = tmp @ P @ self.A

        Fs = []
        pre = np.copy(tmp)
        AcT = (self.A - self.B @ K).T
        for _ in range(self.N):
            Fs.append(pre @ self.C.T * self.q)
            pre = pre @ AcT
        Fs = np.array(Fs).flatten()

        return K, Fs

    def update_state(self, X, U):
        X_next = self.A @ X + self.B @ U
        P_curr = self.C @ X
        return X_next, P_curr

    def solve_system(self, pref, x0 = 0, dx0 = 0, d2x0 = 0):
        '''
            Output:
                Xs: The state vector and ZMP in all sampling time.
                ZMPs: The prediction of ZMPs.
        '''
        # The initial state vector (all zeros by default).
        X = np.array([x0, dx0, d2x0]).T

        n_zmps = len(pref)
        pref_tmp = np.append(pref, [pref[-1]] * (self.N - 1))

        # Go over all samples.
        Xs, pred_ZMPs = np.zeros(n_zmps), np.zeros(n_zmps)
        for i in range(n_zmps):
            U = -np.dot(self.K, X) + np.dot(self.Fs, pref_tmp[i:i + self.N])
            X, ZMP = self.update_state(X, U)
            Xs[i], pred_ZMPs[i] = X[0], ZMP

        return Xs, pred_ZMPs

    def generate(self, N_sup = 4):
        ref_ZMPs, lfoot_traj, rfoot_traj = self.create_ZMP_pattern(N_sup)
        CoMx, pred_ZMPx = self.solve_system(ref_ZMPs[:, 0])
        CoMy, pred_ZMPy = self.solve_system(ref_ZMPs[:, 1])
        
        CoMs = np.stack([CoMx, CoMy, np.full_like(CoMx, self.zc)], axis = 1)
        pred_ZMPs = np.stack([pred_ZMPx, pred_ZMPy, np.zeros_like(pred_ZMPx)], axis = 1)

        return CoMs, pred_ZMPs, ref_ZMPs, lfoot_traj, rfoot_traj


if __name__ == '__main__':
    generator = ZMPWalkPatternGenerator()
    CoMs, pred_ZMPs, ref_ZMPs, lfoot_traj, rfoot_traj = generator.generate(8)

    # ts = np.arange(0, ref_ZMPs.shape[0])
    # plt.plot(ts, ref_ZMPs[:, 0], label = 'Pred ZMP X')
    # plt.plot(ts, pred_ZMPs[:, 0], label = 'Ref ZMP X')
    # plt.plot(ts, CoMs[:, 0], label = 'CoM X')
    # plt.legend()
    # # plt.savefig('x.jpg')
    # plt.show()

    # plt.plot(ts, ref_ZMPs[:, 1], label = 'Pred ZMP Y')
    # plt.plot(ts, pred_ZMPs[:, 1], label = 'Ref ZMP Y')
    # plt.plot(ts, CoMs[:, 1], label = 'CoM Y')
    # plt.legend()
    # # plt.savefig('y.jpg')
    # plt.show()

    # plt.plot(CoMs[:, 0], CoMs[:, 1])
    # # plt.savefig('CoM.jpg')
    # plt.show()

    # plt.plot(pred_ZMPs[:, 0], pred_ZMPs[:, 1])
    # # plt.savefig('pred_ZMP.jpg')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')

    # ax.scatter(CoMs[:, 0], CoMs[:, 1], CoMs[:, 2], label = 'CoM')
    # ax.scatter(lfoot_traj[:, 0], lfoot_traj[:, 1], lfoot_traj[:, 2], label = 'LF')
    # ax.scatter(rfoot_traj[:, 0], rfoot_traj[:, 1], rfoot_traj[:, 2], label = 'RF')
    # plt.legend()
    # plt.show()