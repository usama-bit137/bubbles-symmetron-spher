import numpy as np 

#Potential properties:
def potential(x, w, r): 
    return (1/4)*(x**2-(1-r))**2 + 0.5*w*(x-1)

def potential_derivative(x,w,r):
    return x*(x**2-(1-r)) + 0.5*w

def potential_corr(x, E, r):
    x_plus = Newton_Raphson(1, E, r, 100)
    return potential(x, E, r) - potential(x_plus, E, r)

def potential_sec_der(x, w, r):
    return 3*x**2-(1-r) 

#Root-finding:
def Newton_Raphson(x, w, r, n):
    for i in range(n):
        x += -potential_derivative(x, w, r) / potential_sec_der(x, w, r)
    return x

# ODE Solution:
def ODE(x, v, t, w, r):
    t_s = 0.1
    if t < t_s: 
        r = 10
    else: 
        r = 0
    return -2*v/t + potential_derivative(x, w, r)

def RK_22(x0, E, r):
    
    t_range = 50
    x_range = 2
    dt = 0.001
    t0 = 1e-15
    v0 = 0  
    
    # Boxes to fill:
    xh_total = []
    t_total = []
    v_total = []

    while t0 < t_range:
        xh = x0 + dt * v0 / 2
        if abs(x0) > x_range:
            break

        vh = v0 + ODE(x0, v0, t0, E, r) * dt / 2
        x0 += dt * vh
        v0 += dt * ODE(xh, vh, t0 + dt / 2, E, r)
        t0 += dt
    
        # Fill the boxes:
        v_total.append(vh)
        xh_total.append(xh)
        t_total.append(t0)

    return np.array([t_total, xh_total, v_total])

def IntBisec(a_u, a_o, E, r, N):
    for i in range(N):

        Phi_u = RK_22(a_u, E, r)
        amid = np.float128(0.5 * (a_u + a_o))

        Phi_mid = RK_22(amid, E, r)

        
        #testing the tolerance of the solution:
        if abs(Phi_u[0, -1] - Phi_mid[0, -1]) < 0.000001:
            a_u = np.float128(amid)
        else:
            a_o = np.float128(amid)
    return amid

