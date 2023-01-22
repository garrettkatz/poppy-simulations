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

if __name__ == "__main__":

    # x1, y1, r1 = -5, 1, 4
    # x2, y2, r2 = -4, 5, 3
    # x, y = intersect_circles(x1, y1, r1, x2, y2, r2)    
    # pt.gca().add_patch(pt.Circle((x1, y1), r1, fc='none', ec='b'))
    # pt.gca().add_patch(pt.Circle((x2, y2), r2, fc='none', ec='b'))
    # pt.plot(x, y, 'ko')
    # pt.show()    

    lengths = [.5, .2, 4, 2, 2, .2, .5]

    al = .05*np.pi # half-angle between upper legs and y-axis in initial stance

    # frames = np.zeros((2,2,3))
    # frames[0,0,0] = 1
    # frames[1,1,0] = 1
    # frames[1,1,2] = .5

    frames = fk(
        [np.pi, -np.pi*.75, np.pi*.25 + al, np.pi - 2*al, 0, np.pi*1.25 - (np.pi*1.5 - al), np.pi*.75],
        lengths
    )

    draw(frames)
    pt.axis('equal')
    pt.show()

    aa = np.pi*.47 # angle of front leg relative to x-axis
    at = np.pi*.51 # angle of back sole relative to x-axis

    xa, ya = frames[2,:,2] # front ankle
    xw, yw = xa + lengths[2]*np.cos(aa), ya + lengths[2]*np.sin(aa) # waist
    xt, yt = frames[-1,:,2] # back toe
    xh, yh = xt + lengths[-1]*np.cos(at), yt + lengths[-1]*np.sin(at) # back heel
    xb, yb = xh + lengths[-2]*np.cos(at - np.pi*.75), yh + lengths[-2]*np.sin(at - np.pi*.75) # back ankle
    xk, yk = intersect_circles(xb, yb, lengths[4], xw, yw, lengths[3]) # back knee

    # kin chain angles
    angles = [0]*7
    angles[0] = np.pi # front toe
    angles[1] = -np.pi*.75 # front heel
    angles[2] = aa - sum(angles[:2]) # front ankle
    angles[3] = np.arctan2(yk-yw, xk-xw) - aa # waist
    angles[4] = np.arctan2(yb-yk, xb-xk) - sum(angles[:4]) # back knee
    angles[5] = np.arctan2(yh-yb, xh-xb) - sum(angles[:5]) # back ankle
    angles[6] = np.pi*.75 # back heel

    frames = fk(angles, lengths)
    print(frames[-1,:,2])
    
    draw(frames)
    pt.axis('equal')
    pt.show()


