def draw():

    """
    simple plotting routine
    """
    plt.clf()

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ct = ax1.contourf(x, y, u[:,:,int(N/2)], 100)
    ct = ax2.contourf(x, y, v[:,:,int(N/2)], 100)
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
    u = np.fft.ifftn(u_hat)
    v = np.fft.ifftn(v_hat)

    f = -u*v*v + feed*(1 - u)
    g = u*v*v - (feed + kill)*v

    f_hat = np.fft.fftn(f)
    g_hat = np.fft.fftn(g)

    return f_hat, g_hat

def euler(u_hat, v_hat, int_fac_u2, int_fac_v2):
    k_hat_1, l_hat_1 = rhs_hat(u_hat, v_hat)
    k_hat_1 *= dt; l_hat_1 *= dt
    u_hat = u_hat + k_hat_1 * int_fac_u2
    v_hat = v_hat + l_hat_1 * int_fac_v2
    return u_hat, v_hat

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
# from mayavi.mlab import contour3d

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic_r'

plot = True
if plot:
    fig = plt.figure(figsize=[8, 4])
    
#number of gridpoints in 1D for reference model (HF model)
I = 7
N = 2**I

#domain size [-L, L]
L = 1.25

#2D grid, scaled by L
x = (2*L/N)*np.arange(-N/2, N/2); y=x; z=x
xx, yy, zz = np.meshgrid(x, y, z)

#frequencies of fftn
k = np.fft.fftfreq(N)*N
#frequencies must be scaled as well
k = k * np.pi/L
kx = np.zeros([N, N, N]) + 0.0j
ky = np.zeros([N, N, N]) + 0.0j
kz = np.zeros([N, N, N]) + 0.0j

for i1 in range(N):
    for i2 in range(N):
        for i3 in range(N):
            kx[i1, i2, i3] = 1j*k[i1]
            ky[i1, i2, i3] = 1j*k[i2]
            kz[i1, i2, i3] = 1j*k[i3]

#frequencies of fft2 (only used to compute the spectra)
k_squared = kx**2 + ky**2 + kz**2

#diffusion coefficients
epsilon_u = 2e-3
epsilon_v = 1e-3

feed = 0.02
kill = 0.05

#time step parameters
dt = 0.5
n_steps = 3000
plot_frame_rate = 10

#Initial condition
common_exp = np.exp(-10*(xx**2/2 + yy**2 + zz**2)) + \
             np.exp(-50*((xx-0.5)**2 + (yy-0.5)**2 + zz**2))
u = 1 - 0.5 * common_exp
v = 0.25 * common_exp
# u = u + np.random.rand(N, N, N)*0.01
# v = v + np.random.rand(N, N, N)*0.01

#original Pearson IC
# u = np.ones([N, N, N])
# v = np.zeros([N, N, N])
# left = int(N/2)-10
# right = int(N/2)+10

# u[left:right, left:right, left:right] = 0.33 
# v[left:right, left:right, left:right] = 1/4 
# u[left:right, left:right, left:right] += np.random.rand(20,20,20)*0.01
# v[left:right, left:right, left:right] += np.random.rand(20,20,20)*0.01 
# u += np.random.rand(N,N,N)*0.01
# u += np.random.rand(N,N,N)*0.01

u_hat = np.fft.fftn(u)
v_hat = np.fft.fftn(v)

#Integrating factors
int_fac_u = np.exp(epsilon_u * k_squared * dt / 2)
int_fac_u2 = np.exp(epsilon_u * k_squared * dt)
int_fac_v = np.exp(epsilon_v * k_squared * dt / 2)
int_fac_v2 = np.exp(epsilon_v * k_squared * dt)

#counter for plotting
j = 0

#time stepping
for n in range(n_steps):

    # u_hat, v_hat = rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2)
    u_hat, v_hat = euler(u_hat, v_hat, int_fac_u2, int_fac_v2)

    j += 1
    #plot while running simulation
    if j == plot_frame_rate and plot == True:
        j = 0
        u = np.fft.ifftn(u_hat)
        v = np.fft.ifftn(v_hat)
        draw()

plt.show()