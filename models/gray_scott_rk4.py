def draw():

    """
    simple plotting routine
    """
    plt.clf()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ct = ax1.contourf(x, y, u, 100)
    ct = ax2.contourf(x, y, v, 100)
    plt.tight_layout()

    plt.pause(0.1)

def rhs_hat(u_hat, v_hat):
    """    
    Right hand side of the 2D Gray-Scott equations

    Parameters
    ----------
    u_hat : Fourier coefficients of u
    v_hat : Fourier coefficients of v

    Returns
    -------
    The Fourier coefficients of the right-hand side of u and v (f_hat & g_hat)

    """
    u = np.fft.irfft2(u_hat)
    v = np.fft.irfft2(v_hat)

    f = -u*v*v + feed*(1 - u)
    g = u*v*v - (feed + kill)*v

    f_hat = np.fft.rfft2(f)
    g_hat = np.fft.rfft2(g)

    return f_hat, g_hat

def rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2):
    """
    Runge-Kutta 4 time-stepping subroutine

    Parameters
    ----------
    u_hat : Fourier coefficients of u
    v_hat : Fourier coefficients of v

    Returns
    -------
    u_hat and v_hat at the next time step

    """
    #RK4 step 1
    k_hat_1, l_hat_1 = rhs_hat(u_hat, v_hat)
    k_hat_1 *= dt; l_hat_1 *= dt
    u_hat_2 = (u_hat + k_hat_1 / 2) * int_fac_u
    v_hat_2 = (v_hat + l_hat_1 / 2) * int_fac_v
    #RK4 step 2
    k_hat_2, l_hat_2 = rhs_hat(u_hat_2, v_hat_2)
    k_hat_2 *= dt; l_hat_2 *= dt
    u_hat_3 = u_hat*int_fac_u + k_hat_2 / 2
    v_hat_3 = v_hat*int_fac_v + l_hat_2 / 2
    #RK4 step 3
    k_hat_3, l_hat_3 = rhs_hat(u_hat_3, v_hat_3)
    k_hat_3 *= dt; l_hat_3 *= dt
    u_hat_4 = u_hat * int_fac_u2 + k_hat_3 * int_fac_u
    v_hat_4 = v_hat * int_fac_v2 + l_hat_3 * int_fac_v
    #RK4 step 4
    k_hat_4, l_hat_4 = rhs_hat(u_hat_4, v_hat_4)
    k_hat_4 *= dt; l_hat_4 *= dt
    u_hat = u_hat * int_fac_u2 + 1/6 * (k_hat_1 * int_fac_u2 + 
                                        2 * k_hat_2 * int_fac_u + 
                                        2 * k_hat_3 * int_fac_u + 
                                        k_hat_4)
    v_hat = v_hat * int_fac_v2 + 1/6 * (l_hat_1 * int_fac_v2 + 
                                        2 * l_hat_2 * int_fac_v + 
                                        2 * l_hat_3 * int_fac_v + 
                                        l_hat_4)
    return u_hat, v_hat

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic_r'

plot = True
if plot:
    fig = plt.figure(figsize=[8, 4])

#number of gridpoints in 1D for reference model (HF model)
I = 8
N = 2**I

#domain size [-L, L]
L = 1.25

#2D grid, scaled by L
x = (2*L/N)*np.arange(-N/2, N/2); y=x
xx, yy = np.meshgrid(x, y)

#frequencies of rfft2
k = np.fft.fftfreq(N)*N
#frequencies must be scaled as well
k = k * np.pi/L
kx = np.zeros([N, int(N/2+1)]) + 0.0j
ky = np.zeros([N, int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

#frequencies of fft2 (only used to compute the spectra)
k_squared = kx**2 + ky**2

#diffusion coefficients
epsilon_u = 2e-5
epsilon_v = 1e-5

#alpha pattern
feed = 0.02
kill = 0.055

#beta pattern
# feed = 0.02
# kill = 0.045

#epsilon pattern
# feed = 0.02
# kill = 0.055

#time step parameters
dt = 1
n_steps = 3001
plot_frame_rate = 100

#Initial condition
common_exp = np.exp(-10*(xx**2/2 + yy**2)) + np.exp(-50*((xx-0.5)**2 + (yy-0.5)**2))
u = 1 - 0.5 * common_exp
v = 0.25 * common_exp
# u = u + np.random.rand(N, N)*0.01
# v = v + np.random.rand(N, N)*0.01

#original Pearson IC
# u = np.ones([N, N])
# v = np.zeros([N, N])
# u[int(N/2)-10:int(N/2)+10, int(N/2)-10:int(N/2)+10] = 1/2 
# v[int(N/2)-10:int(N/2)+10, int(N/2)-10:int(N/2)+10] = 1/4 
# u += np.random.rand(N, N) * 0.1
# v += np.random.rand(N, N) * 0.1

u_hat = np.fft.rfft2(u)
v_hat = np.fft.rfft2(v)

#Integrating factors
int_fac_u = np.exp(epsilon_u * k_squared * dt / 2)
int_fac_u2 = np.exp(epsilon_u * k_squared * dt)
int_fac_v = np.exp(epsilon_v * k_squared * dt / 2)
int_fac_v2 = np.exp(epsilon_v * k_squared * dt)

#counter for plotting
j = 0

#time stepping
for n in range(n_steps):

    u_hat, v_hat = rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2)

    j += 1
    #plot while running simulation
    if j == plot_frame_rate and plot == True:
        j = 0
        u = np.fft.irfft2(u_hat)
        v = np.fft.irfft2(v_hat)
        draw()

fig = plt.figure(figsize=[4,4])
ax = fig.add_subplot(111, xlabel=r'$x$', ylabel=r'$y$', title='t = %.1f' % (n*dt))
ct = ax.contourf(xx, yy, np.fft.irfft2(u_hat), 100)
plt.tight_layout()

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(ct, cax=cax)

# plt.axis('scaled')
# ax.plot_surface(xx, yy, np.fft.irfft2(v_hat), cmap='seismic')
# plt.tight_layout()

plt.show()