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
    frames = np.empty((len(angles)+1, 2, 3))
    frames[0] = np.eye(2,3)
    for k in range(len(angles)):
        s, c = np.sin(angles[k]), np.cos(angles[k])
        R = np.array([[c, -s], [s, c]])
        frames[k+1,:,:2] = frames[k,:,:2] @ R
        frames[k+1,:,2:] = frames[k+1,:,:2] @ np.array([[lengths[k], 0]]).T + frames[k,:,2:]
    return frames

def intersect_circles(x1, y1, r1, x2, y2, r2):
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    d1 = (d**2 - r2**2 + r1**2) / (2*d)

    psi = np.arctan2(y2-y1, x2-x1)
    phi = np.arccos(d1/r1)
    ang = psi - phi
    
    x, y = x1 + r1*np.cos(ang), y1 + r1*np.sin(ang)
    return x, y

def stance_ik(a0, a4, x4, y4, lengths):
    x1, y1 = lengths[0]*np.cos(a0), lengths[0]*np.sin(a0)
    x3, y3 = x4 + lengths[-1]*np.cos(a4), y4 + lengths[-1]*np.sin(a4)
    x2, y2 = intersect_circles(x3, y3, lengths[2], x1, y1, lengths[1])
    a1 = np.arctan2(y2-y1, x2-x1) - a0
    a2 = np.arctan2(y3-y2, x3-x2) - (a1 + a0)
    a3 = np.arctan2(y4-y3, x4-x3) - (a2 + a1 + a0)
    return a1, a2, a3

if __name__ == "__main__":

    # x1, y1, r1 = -5, 1, 4
    # x2, y2, r2 = -4, 5, 3
    # x, y = intersect_circles(x1, y1, r1, x2, y2, r2)    
    # pt.gca().add_patch(pt.Circle((x1, y1), r1, fc='none', ec='b'))
    # pt.gca().add_patch(pt.Circle((x2, y2), r2, fc='none', ec='b'))
    # pt.plot(x, y, 'ko')
    # pt.show()    

    lengths = [4, 2, 2, .5]

    # frames = np.zeros((2,2,3))
    # frames[0,0,0] = 1
    # frames[1,1,0] = 1
    # frames[1,1,2] = .5

    frames = fk(
        [np.pi*.51, -np.pi*1.02, 0, np.pi*.51],
        lengths
    )

    draw(frames)
    pt.axis('equal')
    pt.show()

    x4, y4 = frames[4,:,2]
    print(x4, y4)
    a0 = np.pi*.49
    a4 = np.pi*.95
    a1, a2, a3 = stance_ik(a0, a4, x4, y4, lengths)
    

    frames = fk(
        [a0, a1, a2, a3],
        lengths
    )
    print(frames[4,:,2])
    
    draw(frames)
    pt.axis('equal')
    pt.show()


