def get_B_hat_n(N1_hat_n, N2_hat_n, P):
    
    N1_n = np.fft.ifft2(N1_hat_n)
    N2_n = np.fft.ifft2(N2_hat_n)
    
    B1_n = -N1_n**2 - 0.5*N1_n*N2_n
    B2_n = -N2_n**2 - 0.5*N1_n*N2_n
    
    B1_hat_n = P*np.fft.fft2(B1_n)
    B2_hat_n = P*np.fft.fft2(B2_n)
    
    return B1_hat_n, B2_hat_n

def get_N_hat_np1(N1_hat_n, N1_hat_nm1, N2_hat_n, N2_hat_nm1, norm_factor, P):

    B1_hat_n, B2_hat_n = get_B_hat_n(N1_hat_n, N2_hat_n, P)

    N1_hat_np1 = norm_factor*P*(2.0/dt*N1_hat_n - 1.0/(2.0*dt)*N1_hat_nm1 + B1_hat_n)
    N2_hat_np1 = norm_factor*P*(2.0/dt*N2_hat_n - 1.0/(2.0*dt)*N2_hat_nm1 + B2_hat_n)

    return N1_hat_np1, N2_hat_np1

#compute spectral filter
def get_P(cutoff, N):
    
    #frequencies of fft2
    k = np.fft.fftfreq(N)*N
    kx = np.zeros([N, N]) + 0.0j
    ky = np.zeros([N, N]) + 0.0j
    
    for i in range(N):
        for j in range(N):
            kx[i, j] = 1j*k[j]
            ky[i, j] = 1j*k[i]
    
    P = np.ones([N, N])
    
    for i in range(N):
        for j in range(N):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P, k, kx, ky

def draw():
    
    """
    simple plotting routine
    """
    
    plt.clf()

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.plot_surface(x_HF, y_HF, N1_np1_HF)
    ax2.plot_surface(x_HF, y_HF, N2_np1_HF)
    plt.tight_layout()
    
    plt.pause(0.5)

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
# from drawnow import drawnow

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

plot = True
if plot:
    fig = plt.figure(figsize=[8, 4])

#number of gridpoints in 1D for reference model (HF model)
I = 8
N = 2**I

#number of gridpoints in 1D for low resolution model, denoted by _LF
N_LF = 2**(I-2)
N_LF_squared = N_LF**2

#2D grid
axis_LF = np.linspace(0, 2.0*np.pi, N_LF)
x_LF , y_LF = np.meshgrid(axis_LF, axis_LF)
axis_HF = np.linspace(0, 2.0*np.pi, N)
x_HF , y_HF = np.meshgrid(axis_HF, axis_HF)

#cutoff in pseudospectral method
Ncutoff = np.int(N/3)           #reference cutoff
Ncutoff_LF = np.int(N_LF/3)     #cutoff of low resolution (LF) model

#standard spectral filters
P, k, k_x, k_y = get_P(Ncutoff, N)                      #HF model
P_HF2LF, _, _, _ = get_P(Ncutoff_LF, N)                 #HF model project to LF grid
P_LF, k_LF, k_x_LF, k_y_LF = get_P(Ncutoff_LF, N_LF)    #LF model

#k^2, used to compute the Laplace operator
k_squared_LF = k_x_LF**2 + k_y_LF**2            #LF model case
k_squared_no_zero_LF = np.copy(k_squared_LF)
k_squared_no_zero_LF[0,0] = 1.0

k_squared = k_x**2 + k_y**2                     #HF model case
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

kx_full = np.zeros([N_LF, N_LF]) + 0.0j
ky_full = np.zeros([N_LF, N_LF]) + 0.0j

for i in range(N_LF):
    for j in range(N_LF):
        kx_full[i, j] = 1j*k_LF[j]
        ky_full[i, j] = 1j*k_LF[i]

k_squared_full = kx_full**2 + ky_full**2

#start, end time, end time of, time step
dt = 0.001
t = 0.0
t_end = 100
n_steps = np.int(np.round((t_end-t)/dt))

plot_frame_rate = np.floor(0.5/dt).astype('int')

#Create initial conditions
N1 = np.zeros([N, N])
N2 = np.ones([N, N])
#indices of center square
idx1, idx2 = np.where((x_HF > 2*np.pi/3)*(x_HF < 4*np.pi/3)*
                      (y_HF > 2*np.pi/3)*(y_HF < 4*np.pi/3))
#IC values for N1 and N2
N1[idx1, idx2] = 1.0
N2[idx1, idx2] = 0.0

#IC fourier coefs
N1_hat_n_HF = np.fft.fft2(N1)
N2_hat_n_HF = np.fft.fft2(N2)
N1_hat_nm1_HF = np.fft.fft2(N1)
N2_hat_nm1_HF = np.fft.fft2(N2)

nu = 0.01

j = 0

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared - 1)        #for reference solution

for n in range(n_steps):
    N1_hat_np1_HF, N2_hat_np1_HF = get_N_hat_np1(N1_hat_n_HF, N1_hat_nm1_HF, 
                                                N2_hat_n_HF, N2_hat_nm1_HF,
                                                norm_factor, P)
    
    j += 1 
    #plot solution every plot_frame_rate. 
    if j == plot_frame_rate and plot == True:
        j = 0

        N1_np1_HF = np.fft.ifft2(N1_hat_np1_HF).real
        N2_np1_HF = np.fft.ifft2(N2_hat_np1_HF).real
        draw()

    N1_hat_nm1_HF = N1_hat_n_HF
    N1_hat_n_HF = N1_hat_np1_HF
    N2_hat_nm1_HF = N2_hat_n_HF
    N2_hat_n_HF = N2_hat_np1_HF

N1_np1_HF = np.fft.ifft2(N1_hat_np1_HF).real
N2_np1_HF = np.fft.ifft2(N2_hat_np1_HF).real

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.plot_surface(x_HF, y_HF, N1_np1_HF)
ax2.plot_surface(x_HF, y_HF, N2_np1_HF)

plt.show()
