import numpy as np

def spinor(k, s):                           # |ks> = (1, s e^{i theta})/sqrt2, HHM Eqs. 16-17
    R = np.hypot(np.sin(k[0]), np.sin(k[1]))
    return np.array([1.0, s*(np.sin(k[1]) - 1j*np.sin(k[0]))/R]) / np.sqrt(2)

def uv(k, s, D0, Ds, mu, VSO, t=1.0):       # u, v (v has the sign of Dbar), E, xi, Dbar
    eps = -2*t*(np.cos(k[0]) + np.cos(k[1])) - 2*s*VSO*np.hypot(np.sin(k[0]), np.sin(k[1]))
    xi, Db = eps - mu, D0 - Ds*eps/(4*t)    # Dbar_ks = D0 - Ds eps_ks / 4t  (signed)
    E = np.hypot(xi, Db)
    return np.sqrt((1 + xi/E)/2), np.sign(Db)*np.sqrt((1 - xi/E)/2), E, xi, Db

def M_helicity(k, s, kp, sp, vertex, eta, D0, Ds, mu, VSO, t=1.0):
    """M(ks|k's') = <ks|O|k's'> (u u' - eta v v').
    vertex(k, kp): 2x2 spin matrix of O;  eta = +1 (T-even) or -1 (T-odd)."""
    u,  v  = uv(k,  s,  D0, Ds, mu, VSO, t)[:2]
    up, vp = uv(kp, sp, D0, Ds, mu, VSO, t)[:2]
    B = np.vdot(spinor(k, s), vertex(k, kp) @ spinor(kp, sp))
    return B*(u*up - eta*v*vp)

def vertex_scalar(k, kp):                   # ultrasound: spin independent, eta = +1
    return np.eye(2)

def vertex_current(k, kp, A, VSO, t=1.0):   # EM, q -> 0: A . dH/dk at (k+k')/2, eta = -1
    km = ((k[0] + kp[0])/2, (k[1] + kp[1])/2)
    sx, sy = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])
    Vx = 2*t*np.sin(km[0])*np.eye(2) + 2*VSO*np.cos(km[0])*sy
    Vy = 2*t*np.sin(km[1])*np.eye(2) - 2*VSO*np.cos(km[1])*sx
    return A[0]*Vx + A[1]*Vy
