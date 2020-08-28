def draw():

    """
    simple plotting routine
    """
    plt.clf()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # ct = ax1.contourf(xx, yy, v, 100)
    # plt.colorbar(ct)
    # ct = ax2.contourf(xx_LF, yy_LF, v_LF, 100)
    # plt.colorbar(ct)

    ax1.plot(T, plot_dict_HF[0], label='HF')
    ax1.plot(T, plot_dict_LF[0], label='LF')
    ax1.legend(loc=0)
    ax2.plot(T, plot_dict_HF[1], label='HF')
    ax2.plot(T, plot_dict_LF[1], label='LF')
    ax2.legend(loc=0)
    ax3.plot(T, plot_dict_HF[2], label='HF')
    ax3.plot(T, plot_dict_LF[2], label='LF')
    ax3.legend(loc=0)
    ax4.plot(T, plot_dict_HF[3], label='HF')
    ax4.plot(T, plot_dict_LF[3], label='LF')
    ax4.legend(loc=0)

    plt.tight_layout()

    plt.pause(0.1)
    
def get_grid(N):
    """
    Generate an equidistant N x N square grid

    Parameters
    ----------
    N : number of point in 1 dimension
    
    Returns
    -------
    xx, yy: the N x N coordinates

    """
    x = (2*L/N)*np.arange(-N/2, N/2); y=x
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def get_derivative_operator(N):
    """
    Get the spectral operators used to compute the spatial dervatives in 
    x and y direction

    Parameters
    ----------
    N : number of points in 1 dimension
    
    Returns
    -------
    kx, ky: operators to compute derivatives in spectral space. Already
    multiplied by the imaginary unit 1j

    """
    #frequencies of fft2
    k = np.fft.fftfreq(N)*N
    #frequencies must be scaled as well
    k = k * np.pi/L
    kx = np.zeros([N, N]) + 0.0j
    ky = np.zeros([N, N]) + 0.0j
    
    for i in range(N):
        for j in range(N):
            kx[i, j] = 1j*k[j]
            ky[i, j] = 1j*k[i]

    return kx, ky

def get_spectral_filter(kx, ky, N, cutoff):
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

def initial_cond(xx, yy):
    """
    Compute the initial condition    

    Parameters
    ----------
    xx : spatial grid points in x direction
    yy : spatial grid points in y direction
    
    Returns
    -------
    u_hat, v_hat: initial Fourier coefficients of u and v

    """
    common_exp = np.exp(-10*(xx**2/2 + yy**2)) + \
                 np.exp(-50*((xx-0.5)**2 + (yy-0.5)**2))
    u = 1 - 0.5 * common_exp
    v = 0.25 * common_exp
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    return u_hat, v_hat

def integrating_factors(k_squared):
    """
    Compute the integrating factors used in the RK4 time stepping    

    Parameters
    ----------
    k_squared : the operator to compute the Laplace operator

    Returns
    -------
    The integrating factors for u and v

    """
    
    int_fac_u = np.exp(epsilon_u * k_squared * dt / 2)
    int_fac_u2 = np.exp(epsilon_u * k_squared * dt)
    int_fac_v = np.exp(epsilon_v * k_squared * dt / 2)
    int_fac_v2 = np.exp(epsilon_v * k_squared * dt)
    
    return int_fac_u, int_fac_u2, int_fac_v, int_fac_v2
    
def rhs_hat(u_hat, v_hat, **kwargs):
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
    
    if 'dQ' in kwargs:
        dQ = kwargs['dQ']
        #QoI basis functions V
        V_hat = np.zeros([N_Q, N_LF, N_LF]) + 0.0j
        V_hat[0] = V_hat_1_LF
        V_hat[1] = u_hat
        EF_hat_u, c_ij, inner_prods, src_Q, tau = reduced_r(V_hat, dQ[0:N_Q])
        EF_u = np.fft.ifft2(EF_hat_u)

        V_hat = np.zeros([N_Q, N_LF, N_LF]) + 0.0j
        V_hat[0] = V_hat_1_LF
        V_hat[1] = v_hat
        EF_hat_v, c_ij, inner_prods, src_Q, tau = reduced_r(V_hat, dQ[N_Q:])
        EF_v = np.fft.ifft2(EF_hat_v)
        
    else:
       EF_u = EF_v = 0.0
    
    u = np.fft.ifft2(u_hat)
    v = np.fft.ifft2(v_hat)

    f = -u*v*v - EF_u + feed*(1 - u)
    g = u*v*v - EF_v - (feed + kill)*v

    f_hat = np.fft.fft2(f)
    g_hat = np.fft.fft2(g)

    return f_hat, g_hat

def rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2, **kwargs):
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
    k_hat_1, l_hat_1 = rhs_hat(u_hat, v_hat, **kwargs)
    k_hat_1 *= dt; l_hat_1 *= dt
    u_hat_2 = (u_hat + k_hat_1 / 2) * int_fac_u
    v_hat_2 = (v_hat + l_hat_1 / 2) * int_fac_v
    #RK4 step 2
    k_hat_2, l_hat_2 = rhs_hat(u_hat_2, v_hat_2, **kwargs)
    k_hat_2 *= dt; l_hat_2 *= dt
    u_hat_3 = u_hat*int_fac_u + k_hat_2 / 2
    v_hat_3 = v_hat*int_fac_v + l_hat_2 / 2
    #RK4 step 3
    k_hat_3, l_hat_3 = rhs_hat(u_hat_3, v_hat_3, **kwargs)
    k_hat_3 *= dt; l_hat_3 *= dt
    u_hat_4 = u_hat * int_fac_u2 + k_hat_3 * int_fac_u
    v_hat_4 = v_hat * int_fac_v2 + l_hat_3 * int_fac_v
    #RK4 step 4
    k_hat_4, l_hat_4 = rhs_hat(u_hat_4, v_hat_4, **kwargs)
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

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():

    root = tk.Tk()
    root.withdraw()
    fname = filedialog.asksaveasfilename(initialdir = HOME,
                                         title="Save HFD5 file", 
                                         filetypes=(('HDF5 files', '*.hdf5'), 
                                                    ('All files', '*.*')))
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f_store = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f_store.create_dataset(q, data = samples[q])
        
    h5f_store.close()    

###########################
# REDUCED SGS SUBROUTINES #
###########################

def reduced_r(V_hat, dQ):
    """
    Compute the reduced SGS term
    """
    
    #compute the T_ij basis functions
    T_hat = np.zeros([N_Q, N_Q, N_LF, N_LF]) + 0.0j
    
    for i in range(N_Q):

        T_hat[i, 0] = V_hat[i]
        
        J = np.delete(np.arange(N_Q), i)
        
        idx = 1
        for j in J:
            T_hat[i, idx] = V_hat[j]
            idx += 1

    #compute the coefficients c_ij
    inner_prods = inner_products(V_hat, N_LF)

    c_ij = compute_cij_using_V_hat(V_hat, inner_prods)

    EF_hat = 0.0

    src_Q = np.zeros(N_Q)
    tau = np.zeros(N_Q)

    #loop over all QoI
    for i in range(N_Q):
        #compute the fourier coefs of the P_i
        P_hat_i = T_hat[i, 0]
        for j in range(0, N_Q-1):
            P_hat_i -= c_ij[i, j]*T_hat[i, j+1]
    
        #(V_i, P_i) integral
        src_Q_i = compute_int(V_hat[i], P_hat_i, N_LF)
        
        #compute tau_i = Delta Q_i/ (V_i, P_i)
        tau_i = dQ[i]/src_Q_i        

        src_Q[i] = src_Q_i
        tau[i] = tau_i

        #compute reduced soure term
        EF_hat -= tau_i*P_hat_i
    
    return EF_hat, c_ij, np.triu(inner_prods), src_Q, tau

def compute_cij_using_V_hat(V_hat, inner_prods):
    """
    compute the coefficients c_ij of P_i = T_{i,1} - c_{i,2}*T_{i,2}, - ...
    """

    c_ij = np.zeros([N_Q, N_Q-1])
    
    for i in range(N_Q):
        A = np.zeros([N_Q-1, N_Q-1])
        b = np.zeros(N_Q-1)

        k = np.delete(np.arange(N_Q), i)

        for j1 in range(N_Q-1):
            for j2 in range(j1, N_Q-1):
                A[j1, j2] = inner_prods[k[j1], k[j2]]
                if j1 != j2:
                    A[j2, j1] = A[j1, j2]

        for j1 in range(N_Q-1):
            b[j1] = inner_prods[i, k[j1]]

        if N_Q == 2:
            c_ij[i,:] = b/A
        else:
            c_ij[i,:] = np.linalg.solve(A, b)
            
    return c_ij

def inner_products(V_hat, N):

    """
    Compute all the inner products (V_i, T_{i,j})
    """

    V_hat = V_hat.reshape([N_Q, N_LF_squared])

    return np.dot(V_hat, np.conjugate(V_hat).T)/N**4

def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten()))/N**4 
    return integral.real

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
from tkinter import filedialog
import tkinter as tk
import h5py
import time

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'
HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D for reference model (HF model)
I = 9
N = 2**I
N_LF = 2**(I-2)
N_LF_squared = N_LF**2

#number of time series to track
N_Q = 2

#domain size [-L, L]
L = 1.25

#user flags
plot = True
store = True
compute_ref = False
state_store = True
restart = True

sim_ID = 'alpha_unparam'

if plot == True:
    fig = plt.figure(figsize=[8, 8])
    plot_dict_LF = {}
    plot_dict_HF = {}
    T = []
    for i in range(2*N_Q):
        plot_dict_LF[i] = []
        plot_dict_HF[i] = []
        
#TRAINING DATA SET
QoI = ['Q_HF', 'Q_LF']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['N_LF'] = N_LF
    samples['N'] = N
  
    for q in range(Q):
        samples[QoI[q]] = []

#2D grid, scaled by L
xx, yy = get_grid(N)
xx_LF, yy_LF = get_grid(N_LF)

#spatial derivative operators
kx, ky = get_derivative_operator(N)
kx_LF, ky_LF = get_derivative_operator(N_LF)

#Laplace operator
k_squared = kx**2 + ky**2
k_squared_LF = kx_LF**2 + ky_LF**2

#diffusion coefficients
epsilon_u = 2e-5
epsilon_v = 1e-5

#alpha pattern
feed = 0.02
kill = 0.05

#beta pattern
# feed = 0.02
# kill = 0.045

#epsilon pattern
# feed = 0.02
# kill = 0.055

#time step parameters
dt = 0.1
n_steps = 50000
plot_frame_rate = 100
store_frame_rate = 1
t = 5000.0

#Initial condition
if restart == True:

    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t,1)) + '.hdf5'

    #if fname does not exist, select restart file via GUI
    if os.path.exists(fname) == False:
        root = tk.Tk()
        root.withdraw()
        fname = filedialog.askopenfilename(initialdir = HOME + '/restart',
                                           title="Open restart file", 
                                           filetypes=(('HDF5 files', '*.hdf5'), 
                                                      ('All files', '*.*')))

    #create HDF5 file
    h5f = h5py.File(fname, 'r')

    for key in h5f.keys():
        print(key)
        vars()[key] = h5f[key][:]

    h5f.close()
else:
    u_hat, v_hat = initial_cond(xx, yy)
    u_hat_LF, v_hat_LF = initial_cond(xx_LF, yy_LF)

#load reference statistics from file if the ref model in not executed
if compute_ref == False:
    root = tk.Tk()
    root.withdraw()
    fname = tk.filedialog.askopenfilename(title="Open reference data file")
    h5f = h5py.File(fname, 'r')
    ref_data = h5f['Q_HF'][()]

#Integrating factors
int_fac_u, int_fac_u2, int_fac_v, int_fac_v2 = integrating_factors(k_squared)    
int_fac_u_LF, int_fac_u2_LF, int_fac_v_LF, int_fac_v2_LF = integrating_factors(k_squared_LF)    

#counters
j = 0; j2 = 0

V_hat_1 = np.fft.fft2(np.ones([N, N]))
V_hat_1_LF = np.fft.fft2(np.ones([N_LF, N_LF]))

t0 = time.time()
#time stepping
for n in range(n_steps):

    if np.mod(n, 1000) == 0:
        print('time step %d of %d' %(n, n_steps))

    #reference RK4 solve
    if compute_ref:
        u_hat, v_hat = rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2)
        #compute reference stats
        Q_HF = np.zeros(2*N_Q)
        Q_HF[0] = compute_int(V_hat_1, u_hat, N)
        Q_HF[1] = 0.5*compute_int(u_hat, u_hat, N)
        Q_HF[2] = compute_int(V_hat_1, v_hat, N)
        Q_HF[3] = 0.5*compute_int(v_hat, v_hat, N)

    else:
        #load reference stats from memory
        Q_HF = ref_data[n]   

    #compute LF stats
    Q_LF = np.zeros(2*N_Q)
    Q_LF[0] = compute_int(V_hat_1_LF, u_hat_LF, N_LF)
    Q_LF[1] = 0.5*compute_int(u_hat_LF, u_hat_LF, N_LF)
    Q_LF[2] = compute_int(V_hat_1_LF, v_hat_LF, N_LF)
    Q_LF[3] = 0.5*compute_int(v_hat_LF, v_hat_LF, N_LF)

    dQ = Q_HF - Q_LF    
     
    #Low Fidelity (LF) RK4 solve
    u_hat_LF, v_hat_LF = rk4(u_hat_LF, v_hat_LF, int_fac_u_LF, int_fac_u2_LF, 
                              int_fac_v_LF, int_fac_v2_LF, dQ = np.zeros(4))

    j += 1; j2 += 1; t += dt
    #plot while running simulation
    if j == plot_frame_rate and plot == True:
        j = 0
        # u = np.fft.ifft2(u_hat)
        # v = np.fft.ifft2(v_hat)
        # u_LF = np.fft.ifft2(u_hat_LF)
        # v_LF = np.fft.ifft2(v_hat_LF)
    
        # print('energy_HF u = %.4f, energy_LF u = %.4f' % (Q_HF[0], Q_LF[0]))
        # print('energy_HF v = %.4f, energy_LF v = %.4f' % (Q_HF[1], Q_LF[1]))
        # print('=========================================')
    
        plot_dict_HF[0].append(Q_HF[0])
        plot_dict_HF[1].append(Q_HF[1])
        plot_dict_HF[2].append(Q_HF[2])
        plot_dict_HF[3].append(Q_HF[3])

        plot_dict_LF[0].append(Q_LF[0])
        plot_dict_LF[1].append(Q_LF[1])
        plot_dict_LF[2].append(Q_LF[2])
        plot_dict_LF[3].append(Q_LF[3])
   
        T.append(t)

        draw()

    if j2 == store_frame_rate and store == True:
        j2 = 0
                
        for qoi in QoI:
            samples[qoi].append(eval(qoi))

t1 = time.time()
print('*************************************')
print('Simulation time = %f [s]' % (t1 - t0))
print('*************************************')

#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:

    if compute_ref:
        keys = ['u_hat_LF', 'v_hat_LF', 'u_hat', 'v_hat']
    else:
        keys = ['u_hat_LF', 'v_hat_LF']
   
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')

    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t, 1)) + '.hdf5'
  
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()   

#store the samples
if store == True:
    store_samples_hdf5() 

plt.show()