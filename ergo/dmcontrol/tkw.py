import matplotlib.pyplot as pt
import numpy as np

def draw(frames):
    for k in range(len(frames)):
        a = frames[k,:,2]
        b = a + frames[k,:,0]
        pt.plot([a[0],b[0]], [a[1],b[1]], 'r-')
        if k > 0:
            c = frames[k-1,:,2]
            pt.plot([c[0],a[0]], [c[1],a[1]], 'bo-')

def fk(angles, lengths):
    frames = np.empty((len(angles)+1, 2, 3)
    frames[0] = np.eye((2,3))
    for k in range(len(angles)):
        s, c = np.sin(angles[k]), np.cos(angles[k])
        R = np.array([[c, -s], [s, c]])
        frames[k+1,:,:2] = frames[k,:,:2] * R
        frames[k+1,:,2] = frames[k+1,:,:2] @ np.array([[lengths[k], 0]]).T + frames[k,:,2]

if __name__ == "__main__":

    frames = np.zeros((2,2,3))
    frames[0,0,0] = 1
    frames[1,1,0] = 1
    frames[1,1,2] = .5
    
    draw(frames)
    pt.axis('equal')
    pt.show()

