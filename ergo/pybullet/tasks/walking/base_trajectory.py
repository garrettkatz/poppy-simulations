import numpy as np
import matplotlib.pyplot as pt

def get_traj(y0, y1, dy0, dy1, T):

    D = y0
    C = dy0
    A, B = np.linalg.solve(
        np.array([[1,1],[3,2]]),
        np.stack((y1 - y0 - dy0, dy1 - dy0)))

    s = np.linspace(0, 1, T)[:,np.newaxis]
    t = (1 - np.cos(np.pi * s)) / 2
    Y = A*t**3 + B*t**2 + C*t + D

    # if y0.ndim == 1: y0 = y0[:,np.newaxis]
    # if y1.ndim == 1: y1 = y1[:,np.newaxis]

    # t = np.linspace(0, 1, T)
    # Y = -2*(y1 - y0)*t**3 + 3*(y1 - y0)*t**2 + y0

    return Y.T, s.flatten()

def get_goal():

    heading = np.random.uniform(np.pi/8, np.pi/ 2)
    # alpha = np.random.uniform(heading, np.pi/2)
    alpha = np.random.uniform(np.pi/8, np.pi/2)
    dist = np.random.uniform(1, 4)
    sign = np.random.choice([-1, 1])
    
    y1 = dist * np.array([sign * np.cos(alpha), np.sin(alpha)])
    dy1 = np.array([sign * np.cos(heading), np.sin(heading)])

    return y1, dy1

if __name__ == "__main__":

    y0 = np.array([0, 0])
    dy0 = np.array([0, 1])

    for reps in range(10):

        y1, dy1 = get_goal()
        # y1 = np.array([4, 1])
        # dy1 = np.array([1, 0])

        Y, t = get_traj(y0, y1, dy0, dy1, 100)
        dY = np.linalg.norm((Y[:,1:] - Y[:,:-1]) / (t[1:] - t[:-1]), axis=0)
    
        pt.subplot(2,1,1)
        pt.plot(Y[0], Y[1], '.-')
        pt.xlim([-5, 5])
        pt.ylim([-1, 5])
    
        pt.subplot(2,1,2)
        pt.plot(t[:-1], dY, '.-')

    pt.show()


