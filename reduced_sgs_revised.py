"""
========================================================================
PYTHON SCRIPT ACCOMPANYING:

W.EDELING, D. CROMMELIN, 
"DERIVING REDUCED SUBGRID SCALE MODELS FROM DATA"
COMPUTERS & FLUIDS, 2019.
========================================================================
"""
######################
# SOLVER SUBROUTINES #
######################

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P, kx, ky, k_squared_no_zero):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, kx, ky, k_squared_no_zero, F_hat, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P, kx, ky, k_squared_no_zero)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

#compute spectral filter
def get_P(cutoff, N):
    
    #frequencies of rfft2
    k = np.fft.fftfreq(N)*N
    kx = np.zeros([N, int(N/2+1)]) + 0.0j
    ky = np.zeros([N, int(N/2+1)]) + 0.0j
    
    for i in range(N):
        for j in range(int(N/2+1)):
            kx[i, j] = 1j*k[j]
            ky[i, j] = 1j*k[i]
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P, k, kx, ky

def get_P_k(k_min, k_max, N):
    
    P_k = np.zeros([N, N])    
    idx0, idx1 = np.where((binnumbers >= k_min) & (binnumbers <= k_max))
    
    P_k[idx0, idx1] = 1.0
    
    return P_k[0:N, 0:int(N/2+1)] 

#compute spectral filter
def get_P_full(cutoff, N):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
                P[i, j] = 0.0

    return P

#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n, k_squared_no_zero):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n

##########################
# END SOLVER SUBROUTINES #
##########################

#############################
# MISCELLANEOUS SUBROUTINES #
#############################

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f_store = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f_store.create_dataset(q, data = samples[q])
        
    h5f_store.close()    

def draw():
    
    """
    simple plotting routine
    """
    
    plt.clf()

    plt.subplot(121, title=r'$Q_1$', xlabel=r'$t\;[day]$')
    plt.plot(np.array(T)/day, plot_dict_LF[0], label='Reduced')
    plt.plot(np.array(T)/day, plot_dict_HF[0], 'o', label=r'Reference')
    
#    T_int = np.array([T[0], T[-1]])/day
#    plt.plot(T_int, [e_std, e_std], '--k')
#    plt.plot(T_int, [-e_std, -e_std], '--k')
    
    plt.legend(loc=0)

    plt.subplot(122, title=r'$Q_2$', xlabel=r'$t\;[day]$')
    plt.plot(np.array(T)/day, plot_dict_LF[1])
    plt.plot(np.array(T)/day, plot_dict_HF[1], 'o')
#    plt.plot(T_int, [z_std, z_std], '--k')
#    plt.plot(T_int, [-z_std, -z_std], '--k')
    
    plt.pause(0.05)
    
#plot instantaneous energy spectrum    
#    plt.subplot(122, xscale='log', yscale='log')
#    plt.plot(bins+1, E_spec_HF, '--')
#    plt.plot(bins+1, E_spec_LF)
#    plt.plot([Ncutoff_LF + 1, Ncutoff_LF + 1], [10, 0], 'lightgray')
#    plt.plot([np.sqrt(2)*Ncutoff_LF + 1, np.sqrt(2)*Ncutoff_LF + 1], [10, 0], 'lightgray')

    plt.tight_layout()
    
#################################
# END MISCELLANEOUS SUBROUTINES #
#################################
    
###########################
# REDUCED SGS SUBROUTINES #
###########################

def reduced_r(V_hat, dQ):
    """
    Compute the reduced SGS term
    """
    
    #compute the T_ij basis functions
    T_hat = np.zeros([N_Q, N_Q, N_LF,int(N_LF/2+1)]) + 0.0j
    
    for i in range(N_Q):

        T_hat[i, 0] = V_hat[i]
        
        J = np.delete(np.arange(N_Q), i)
        
        idx = 1
        for j in J:
            T_hat[i, idx] = V_hat[j]
            idx += 1

    #compute the coefficients c_ij for P_i
#    c_ij = compute_cij(T_hat, V_hat)
    
    inner_prods = inner_products(V_hat)
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
        src_Q_i = compute_int(V_hat[i], P_hat_i, axis_LF)
        
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
#            integral = compute_int(V_hat[k[j1]], T_hat[i, 0])
            b[j1] = inner_prods[i, k[j1]]

        if N_Q == 2:
            c_ij[i,:] = b/A
        else:
            c_ij[i,:] = np.linalg.solve(A, b)
            
    return c_ij


def compute_cij(T_hat, V_hat):
    """
    compute the coefficients c_ij of P_i = T_{i,1} - c_{i,2}*T_{i,2}, - ...
    """

    c_ij = np.zeros([N_Q, N_Q-1])

    for i in range(N_Q):
        A = np.zeros([N_Q-1, N_Q-1])
        b = np.zeros(N_Q-1)

        k = np.delete(np.arange(N_Q), i)

        for j1 in range(N_Q-1):
            for j2 in range(N_Q-1):
                integral = compute_int(V_hat[k[j1]], T_hat[i, j2+1], axis_LF)
                A[j1, j2] = integral

        for j1 in range(N_Q-1):
            integral = compute_int(V_hat[k[j1]], T_hat[i, 0], axis_LF)
            b[j1] = integral

        if N_Q == 2:
            c_ij[i,:] = b/A
        else:
            c_ij[i,:] = np.linalg.solve(A, b)
            
    return c_ij

def get_qoi_euclid(w_hat_n, k_squared_no_zero, target, axis):

    """
    compute the Quantity of Interest defined by the string target
    """
    
    w_n = np.fft.irfft2(w_hat_n).flatten()
    
    #energy (psi, omega)/2
    if target == 'e':
        psi_hat_n = w_hat_n/k_squared_no_zero
        psi_hat_n[0,0] = 0.0
        psi_n = np.fft.irfft2(psi_hat_n).flatten()
#        e_n = -0.5*psi_n*w_n
#        return simps(simps(e_n, axis), axis)/(2*np.pi)**2
        return np.dot(-psi_n, w_n)        
    
    #enstrophy (omega, omega)/2
    elif target == 'z':
#        z_n = 0.5*w_n**2
#        return simps(simps(z_n, axis), axis)/(2*np.pi)**2
        return np.dot(w_n, w_n)
    #average vorticity (1, omega)
    elif target == 'w1':
        return simps(simps(w_n, axis), axis)/(2*np.pi)**2
    #higher moment vorticity (omega^2, omega)/3
    elif target == 'w3':
        w3_n = w_n**3/3.0
        return simps(simps(w3_n, axis), axis)/(2*np.pi)**2
    else:
        print(target, 'IS AN UNKNOWN QUANTITY OF INTEREST')
        import sys; sys.exit()

def get_qoi(w_hat_n, k_squared_no_zero, target, axis):

    """
    compute the Quantity of Interest defined by the string target
    """
    
    w_n = np.fft.irfft2(w_hat_n)
    
    #energy (psi, omega)/2
    if target == 'e':
        psi_hat_n = w_hat_n/k_squared_no_zero
        psi_hat_n[0,0] = 0.0
        psi_n = np.fft.irfft2(psi_hat_n)
        e_n = -0.5*psi_n*w_n
        return simps(simps(e_n, axis), axis)/(2*np.pi)**2
    #enstrophy (omega, omega)/2
    elif target == 'z':
        z_n = 0.5*w_n**2
        return simps(simps(z_n, axis), axis)/(2*np.pi)**2
    #average vorticity (1, omega)
    elif target == 'w1':
        return simps(simps(w_n, axis), axis)/(2*np.pi)**2
    #higher moment vorticity (omega^2, omega)/3
    elif target == 'w3':
        w3_n = w_n**3/3.0
        return simps(simps(w3_n, axis), axis)/(2*np.pi)**2
    #(psi, F)/2
    elif target == 'u':
        psi_hat_n = w_hat_n/k_squared_no_zero
        psi_hat_n[0,0] = 0.0
        psi_n = np.fft.irfft2(psi_hat_n)
        return 0.5*simps(simps(psi_n*F_LF, axis), axis)/(2.0*np.pi)**2
    #(w, F)/2
    elif target == 'v':
        return 0.5*simps(simps(w_n*F_LF, axis), axis)/(2.0*np.pi)**2
    else:
        print(target, 'IS AN UNKNOWN QUANTITY OF INTEREST')
        import sys; sys.exit()
    
def inner_products(V_hat):

    """
    Compute all the inner products (V_i, T_{i,j})
    """
    
    inner = np.zeros([N_Q, N_Q])
    
    for i1 in range(N_Q):
        for i2 in range(i1, N_Q):
            inner[i1, i2] = compute_int(V_hat[i1], V_hat[i2], axis_LF)
            
            if i1 != i2:
                inner[i2, i1] = inner[i1, i2]
    
    return inner
    
def compute_int(X1_hat, X2_hat, axis):
    
    """
    Compute integral using Simpsons rule
    """
    
    X1 = np.fft.irfft2(X1_hat)
    X2 = np.fft.irfft2(X2_hat)
    
    return simps(simps(X1*X2, axis), axis)/(2*np.pi)**2
    # return np.dot(X1, X2)

###############################
# END REDUCED SGS SUBROUTINES #
###############################

#########################
## SPECTRUM SUBROUTINES #
#########################

def freq_map(N, N_cutoff):
    """
    Map 2D frequencies to a 1D bin (kx, ky) --> k
    where k = 0, 1, ..., sqrt(2)*Ncutoff
    """
   
    #edges of 1D wavenumber bins
    bins = np.arange(-0.5, np.ceil(2**0.5*Ncutoff)+1)
    #fmap = np.zeros([N,N]).astype('int')
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            #Euclidian distance of frequencies kx and ky
            dist[i, j] = np.sqrt(kx_full[i,j]**2 + ky_full[i,j]**2).imag
                
    #find 1D bin index of dist
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    
    binnumbers -= 1
            
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat, P, N):

    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat
    w_hat_full[map_I, map_J] = np.conjugate(w_hat[I, J])
    w_hat_full *= P
    
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0
    
    E_hat = -0.5*psi_hat_full*np.conjugate(w_hat_full)/N**4
    Z_hat = 0.5*w_hat_full*np.conjugate(w_hat_full)/N**4
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    for i in range(N):
        for j in range(N):
            bin_idx = binnumbers[i, j]
            E_spec[bin_idx] += E_hat[i, j].real
            Z_spec[bin_idx] += Z_hat[i, j].real
            
    return E_spec, Z_spec

#############################
#  END SPECTRUM SUBROUTINES #
#############################

###########################
# M A I N   P R O G R A M #
###########################

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.integrate import simps
import sys
from scipy import stats
import json
from tkinter import filedialog
import tkinter as tk
import time

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D for reference model (HF model)
I = 8
N = 2**I

#number of gridpoints in 1D for low resolution model, denoted by _LF
N_LF = 2**(I-2)

#2D grid
axis_LF = np.linspace(0, 2.0*np.pi, N_LF)
x_LF , y_LF = np.meshgrid(axis_LF, axis_LF)
axis_HF = np.linspace(0, 2.0*np.pi, N)
x_HF , y_HF = np.meshgrid(axis_HF, axis_HF)

#cutoff in pseudospectral method
Ncutoff = np.int(N/3)           #reference cutoff
Ncutoff_LF = np.int(N_LF/3)     #cutoff of low resolution (LF) model

#spectral filters
P, k, k_x, k_y = get_P(Ncutoff, N)
P_HF2LF, _, _, _ = get_P(Ncutoff_LF, N)
P_LF, k_LF, k_x_LF, k_y_LF = get_P(Ncutoff_LF, N_LF)

#spectral filter for the full FFT2 
#P_full = get_P_full(Ncutoff, N)
#P_LF_full = get_P_full(Ncutoff_LF, N_LF)

k_squared_LF = k_x_LF**2 + k_y_LF**2
k_squared_no_zero_LF = np.copy(k_squared_LF)
k_squared_no_zero_LF[0,0] = 1.0

k_squared = k_x**2 + k_y**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

kx_full = np.zeros([N_LF, N_LF]) + 0.0j
ky_full = np.zeros([N_LF, N_LF]) + 0.0j

for i in range(N_LF):
    for j in range(N_LF):
        kx_full[i, j] = 1j*k_LF[j]
        ky_full[i, j] = 1j*k_LF[i]

k_squared_full = kx_full**2 + ky_full**2
k_squared_no_zero_full = np.copy(k_squared_full)
k_squared_no_zero_full[0,0] = 1.0

#read flags from input file
fpath = sys.argv[1]
fp = open(fpath, 'r')

#print the desription of the input file
print(fp.readline())

binnumbers, bins = freq_map(N_LF, Ncutoff_LF)
N_bins = bins.size

###################
# Read input file #
###################
flags = json.loads(fp.readline())
print('*********************')
print('Simulation flags')
print('*********************')

for key in flags.keys():
    vars()[key] = flags[key]
    print(key, '=', flags[key])

N_Q = int(fp.readline())

targets = []
V = []
P_i = []

for i in range(N_Q):
    qoi_i = json.loads(fp.readline())
    targets.append(qoi_i['target'])
    V.append(qoi_i['V_i'])
    k_min = qoi_i['k_min']
    k_max = qoi_i['k_max']
    P_i.append(get_P_k(k_min, k_max, N_LF))
    
print('*********************')

dW3_calc = np.in1d('dW3', targets)

#map from the rfft2 coefficient indices to fft2 coefficient indices
#Use: see compute_E_Z subroutine
shift = np.zeros(N_LF).astype('int')
for i in range(1,N_LF):
    shift[i] = np.int(N_LF-i)
I = range(N_LF);J = range(np.int(N_LF/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time, end time of, time step
dt = 0.01
t = 0.0*day
t_end = t + 10*day
n_steps = np.int(np.round((t_end-t)/dt))

#############
# USER KEYS #
#############

#framerate of storing data, plotting results (1 = every integration time step)
#store_frame_rate = np.floor(1.0*day/dt).astype('int')
store_frame_rate = 1
plot_frame_rate = np.floor(1.0*day/dt).astype('int')
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')

store_ID = sim_ID 
    
###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#TRAINING DATA SET
QoI = ['Q_HF', 'Q_LF', 'dQ', 'c_ij', 'inner_prods', 'src_Q', 
       'tau']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N_LF'] = N_LF
    
    for q in range(Q):
        
        #assume a field contains the string '_hat_'
        if '_hat_' in QoI[q]:
            samples[QoI[q]] = np.zeros([S, N_LF, int(N_LF/2+1)]) + 0.0j
        #a scalar
        else:
#            samples[QoI[q]] = np.zeros(S)
            samples[QoI[q]] = []

#forcing term
F_LF = 2**1.5*np.cos(5*x_LF)*np.cos(5*y_LF);
F_hat_LF = np.fft.rfft2(F_LF);

F_HF = 2**1.5*np.cos(5*x_HF)*np.cos(5*y_HF);
F_hat_HF = np.fft.rfft2(F_HF);

#V_i for Q_i = (1, omega)
V_hat_w1 = P_LF*np.fft.rfft2(np.ones([N_LF, N_LF]))

if restart == True:
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day,1)) + '.hdf5'

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
    
    #initial condition
    w_LF = np.sin(4.0*x_LF)*np.sin(4.0*y_LF) + 0.4*np.cos(3.0*x_LF)*np.cos(3.0*y_LF) + \
           0.3*np.cos(5.0*x_LF)*np.cos(5.0*y_LF) + 0.02*np.sin(x_LF) + 0.02*np.cos(y_LF)

    w_HF = np.sin(4.0*x_HF)*np.sin(4.0*y_HF) + 0.4*np.cos(3.0*x_HF)*np.cos(3.0*y_HF) + \
           0.3*np.cos(5.0*x_HF)*np.cos(5.0*y_HF) + 0.02*np.sin(x_HF) + 0.02*np.cos(y_HF)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w_HF)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    w_hat_n_LF = P_LF*np.fft.rfft2(w_LF)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P, k_x, k_y, k_squared_no_zero)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF, k_x_LF, k_y_LF, k_squared_no_zero_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

#constant factor that appears in AB/BDI2 time stepping scheme   
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)        #for reference solution
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared_LF + mu)  #for Low-Fidelity (LF) or resolved solution

if compute_ref == False:

    root = tk.Tk()
    root.withdraw()
    fname = filedialog.askopenfilename(initialdir = HOME ,
                                        title="Open reference data file", 
                                        filetypes=(('HDF5 files', '*.hdf5'), 
                                                  ('All files', '*.*')))
        
    #create HDF5 file
    h5f = h5py.File(fname, 'r')
    
#some counters
j = 0; j2 = 0; idx = 0;

if plot == True:
    fig = plt.figure(figsize=[8, 4])
    plot_dict_LF = {}
    plot_dict_HF = {}
    T = []
    for i in range(N_Q):
        plot_dict_LF[i] = []
        plot_dict_HF[i] = []

print('*********************')
print('Solving forced dissipative vorticity equations')
print('Ref grid = ', N, 'x', N)
print('Grid = ', N_LF, 'x', N_LF)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')
print('*********************')

t0 = time.time()

#time loop
for n in range(n_steps):
    
    if np.mod(n, np.int(day/dt)) == 0:
        print('n =', n, 'of', n_steps)

    if compute_ref == True:
        
        #solve for next time step
        w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, 
                                                      VgradW_hat_nm1_HF, P, 
                                                      norm_factor,
                                                      k_x, k_y, k_squared_no_zero,
                                                      F_hat_HF)
        
        #exact eddy forcing
#        EF_hat_nm1_exact = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF 
  
    #exact orthogonal pattern surrogate
    if eddy_forcing_type == 'tau_ortho':
        psi_hat_n_LF = get_psi_hat(w_hat_n_LF, k_squared_no_zero_LF)
        
        #to calculate calculate (w^2, w)
        if dW3_calc:
            w_n_LF = np.fft.irfft2(w_hat_n_LF)
            w_hat_n_LF_squared = P_LF*np.fft.rfft2(w_n_LF**2)

        #QoI basis functions V
        V_hat = np.zeros([N_Q, N_LF, int(N_LF/2+1)]) + 0.0j
       
        Q_LF = np.zeros(N_Q)
        Q_HF = np.zeros(N_Q)
        
        for i in range(N_Q):
            V_hat[i] = P_i[i]*eval(V[i])
            
            if eddy_forcing_type == 'tau_ortho':
#                Q_HF[i] = data_eng.h5f[targets[i] + '_n_HF'][n]
                if compute_ref == True:
                    Q_HF[i] = get_qoi(P_HF2LF*w_hat_n_HF, k_squared_no_zero, targets[i], axis_HF)
                else:
                    Q_HF[i] = h5f['Q_HF'][n][i]
                Q_LF[i] = get_qoi(P_i[i]*w_hat_n_LF, k_squared_no_zero_LF, targets[i], axis_LF)
            dQ = Q_HF - Q_LF

        #compute reduced eddy forcing
        EF_hat, c_ij, inner_prods, src_Q, tau = reduced_r(V_hat, dQ)        

    else:
        print('No valid eddy_forcing_type selected')
        sys.exit()
   
    #########################
    #LF solve
    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, 
                                                  VgradW_hat_nm1_LF, P_LF, 
                                                  norm_factor_LF, 
                                                  k_x_LF, k_y_LF, k_squared_no_zero_LF,
                                                  F_hat_LF, EF_hat)
    t += dt
    j += 1
    j2 += 1    
    
    #plot solution every plot_frame_rate. 
    if j == plot_frame_rate and plot == True:
        j = 0

        for i in range(N_Q):       
            plot_dict_LF[i].append(Q_LF[i])
            plot_dict_HF[i].append(Q_HF[i])
        
        T.append(t)
#        
#        E_spec_HF, Z_spec_HF = spectrum(w_hat_n_HF, P_full)
#        E_spec_LF, Z_spec_LF = spectrum(w_hat_n_LF, P_LF_full)
        
        draw()
        
    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0

        for qoi in QoI:
            samples[qoi].append(eval(qoi))

#        idx += 1  

    #update variables
    if compute_ref == True: 
        w_hat_nm1_HF = np.copy(w_hat_n_HF)
        w_hat_n_HF = np.copy(w_hat_np1_HF)
        VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

t1 = time.time()

print('Simulation time =', t1 - t0, 'seconds')
    
####################################

#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:
    
#    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
#            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']

    keys = ['w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    #cPickle.dump(state, open(HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.pickle', 'w'))
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()   

####################################

#store the samples
if store == True:
    store_samples_hdf5() 

plt.show()
