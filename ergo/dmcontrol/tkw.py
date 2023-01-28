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

def intersect_circles(x1, y1, r1, x2, y2, r2, sign=+1):
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    d1 = (d**2 - r2**2 + r1**2) / (2*d)

    psi = np.arctan2(y2-y1, x2-x1)
    phi = np.arccos(d1/r1)
    ang = psi - sign*phi
    
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

    al = .015*np.pi # half-angle between upper legs and y-axis in initial stance

    init_angles = [np.pi, -np.pi*.75, np.pi*.25 + al, np.pi - 2*al, 0, np.pi*1.25 - (np.pi*1.5 - al), np.pi*.75]
    init_frames = fk(init_angles, lengths)

    pt.subplot(1,3,1)
    draw(init_frames)
    pt.axis('equal')
    # pt.show()

    aa = np.pi*.5 # angle of front leg relative to x-axis
    at = np.pi*.85 # angle of back sole relative to x-axis

    xa, ya = init_frames[2,:,2] # front ankle
    xw, yw = xa + lengths[2]*np.cos(aa), ya + lengths[2]*np.sin(aa) # waist
    xt, yt = init_frames[-1,:,2] # back toe
    xh, yh = xt + lengths[-1]*np.cos(at), yt + lengths[-1]*np.sin(at) # back heel
    xb, yb = xh + lengths[-2]*np.cos(at - np.pi*.75), yh + lengths[-2]*np.sin(at - np.pi*.75) # back ankle
    xk, yk = intersect_circles(xb, yb, lengths[4], xw, yw, lengths[3]) # back knee

    # kin chain angles
    push_angles = [0]*7
    push_angles[0] = np.pi # front toe
    push_angles[1] = -np.pi*.75 # front heel
    push_angles[2] = aa - sum(push_angles[:2]) # front ankle
    push_angles[3] = np.arctan2(yk-yw, xk-xw) - aa # waist
    push_angles[4] = np.arctan2(yb-yk, xb-xk) - sum(push_angles[:4]) # back knee
    push_angles[5] = np.arctan2(yh-yb, xh-xb) - sum(push_angles[:5]) # back ankle
    push_angles[6] = np.pi*.75 # back heel

    push_frames = fk(push_angles, lengths)
    print(f"kf = {yk}")
    
    pt.subplot(1,3,2)
    draw(push_frames)
    pt.axis('equal')
    # pt.show()

    # land pose
    aa = np.pi*.5 # angle of initially front leg relative to x-axis
    ah = np.pi*0.1 # angle between ground and previously back sole after swing landing in front

    # distance from front heel to back toe in init stance
    xfh, yfh = init_frames[1,:,2]
    xbt, ybt = init_frames[-1,:,2]
    dth = xfh - xbt
    xbh, ybh = dth, 0 # previously back heel after swing landing in front
    xba, yba = xbh + lengths[-2]*np.cos(ah + np.pi*.25), ybh + lengths[-2]*np.sin(ah + np.pi*.25) # previously back ankle after swing
    xfa, yfa = init_frames[2,:,2] # previously front ankle
    xw, yw = xfa + lengths[2]*np.cos(aa), yfa + lengths[2]*np.sin(aa) # waist
    xk, yk = intersect_circles(xba, yba, lengths[4], xw, yw, lengths[3]) # previously back knee

    land_angles = [0]*7
    land_angles[0] = np.pi # previously front toe
    land_angles[1] = -np.pi*.75 # previously front heel
    land_angles[2] = np.arctan2(yw-yfa, xw-xfa) - sum(land_angles[:2]) # previously front ankle
    land_angles[3] = np.arctan2(yk-yw, xk-xw) - sum(land_angles[:3]) # waist
    land_angles[4] = np.arctan2(yba-yk, xba-xk) - sum(land_angles[:4]) # previously back knee
    land_angles[5] = np.arctan2(ybh-yba, xbh-xba) - sum(land_angles[:5]) # previously back ankle
    land_angles[6] = np.pi*.75 # previously back heel

    land_frames = fk(land_angles, lengths)
    xbt, ybt = land_frames[-1,:,2]

    # compare distance of knee to floor, heel, and toe
    print(f"kf = {yk}")
    print(f"kh = {np.sqrt((xk-xbh)**2 + (yk-ybh)**2)}")
    print(f"kt = {np.sqrt((xk-xbt)**2 + (yk-ybt)**2)}")

    pt.subplot(1,3,3)
    draw(land_frames)
    pt.axis('equal')
    pt.show()
