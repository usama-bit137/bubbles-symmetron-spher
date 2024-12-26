import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import seaborn as sns

from factory import * 

fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig1.tight_layout()

sns.set_theme(style = "white")

def analytical_approx(t, t_s, r_s, w):
    array = np.zeros(len(t))
    m_0 = 1
    phi_f = Newton_Raphson(1, w, 0, 100)
    ax1.axvline(1/m_0, 
                label="$\lambda_0 = (\sqrt{2}\mu)^{-1}$", 
                linestyle= "dotted", 
                color="darkgreen")
    
    ax1.axvline(t_s, 
                label="$r_s=$" + str(0.1)+ "$\lambda_0$", 
                linestyle= "dotted", 
                color = "mediumvioletred")
    
    prefactor = w*(0.5/r_s + (m_0)**(-2))+ phi_f

    for i in range(len(t)):
        if t[i] < t_s:
            C = (1/(t_s*np.sqrt(r_s)))*prefactor*(1/np.cosh(np.sqrt(r_s)*t_s))
            array[i] = C*(t_s/t[i])*np.sinh(np.sqrt(r_s)*t[i])-w/(2*r_s)
        else:
            K = -prefactor*(1-(1/(t_s*np.sqrt(r_s))*np.tanh(np.sqrt(r_s)*t_s)))
            array[i] = K*(t_s/t[i])*np.exp(-m_0**(-1)*(t[i]-t_s)) + phi_f

    return array

# Fundamental values we require:
N = 1
E = np.linspace(0.01, 0.09, N)
r = 0

# Initial conditions for the Runge-Kutta algorithm.
for i in range(N):        
    a_u = np.float128(+0.1)
    a_o = np.float128(Newton_Raphson(0.9, E[i], r, 1))
    phi_0 = np.float128(Newton_Raphson(1, E[i], r, 20))
    
    a_mid = IntBisec(a_u, a_o, E[i], r, 100)
    phi_mid = RK_22(a_mid, E[i], r)
    x_pot = np.linspace(-1.5,1.5,100)

    t = phi_mid[0]
    x = phi_mid[1]
    v = phi_mid[-1]
    
    #search algorithm: 
    n = 0
    for j in range(len(t)): 
        if abs(t[j] - 30) < 0.000001: 
            n=j
            break

    t_cut = t[:25000]
    x_cut = x[:25000]
    v_cut = v[:25000]
        
    ax1.plot(t_cut, 
             x_cut, 
             color="orange", 
             label="numerical")
    
    ax1.plot(t_cut, analytical_approx(t_cut, 0.1, 10, E[i]), 
             linestyle="dashed", 
             color="k", 
             label="analytical")
    
    ax1.axhline(phi_0, 
                linestyle="-.", 
                color="r", 
                label= "$\phi_f$")
    
    ax2.plot(x_cut, 0.5*v_cut**2)
    ax2.set_xlabel("$\phi$")
    ax2.set_ylabel("V($\phi$)")



"""___________________________________Styles_______________________________"""
s=11
ax1.set_ylim(0.94, 1.025)
ax1.set_xlim(0, 3)
ax1.tick_params(axis='both', 
                which='major', 
                labelsize=s)

ax1.set_title(r"Background field with $\epsilon$ = " + str(np.round(E[i], 3))+ r"$\lambda, \rho_s = $" + str(10) + r'$\mu^2M^2$ and $r_s = $' + str(0.1) + r"$\lambda_0$")
ax1.set_xlabel('$r/\lambda_0$', 
               size=20)
ax1.set_ylabel('$\phi/\phi_0$', 
               size=20)
ax1.legend(loc='upper right', 
           fontsize=s)

#always have this: 
plt.show()