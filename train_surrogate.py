def animate():

#    I = 2
#    for i in range(I):
#        #KVM animation
#        lines[2*i].set_data(dom[:, i], kde[s, :, i]/np.max(kde[s, :, i]))
#        lines[2*i + 1].set_data(y_train[s, i], 0.0)
        
#    #Quantized softmax animation
#    idx1 = np.where(data_eng.binned_data[s] == 1.0)[0]

    ims.append((ax1.vlines(np.arange(n_bins), ymin=np.zeros(n_bins), ymax=o_i[0]),
                ax1.plot(idx1[0], 0, 'ro')[0],))
    ims.append((ax2.vlines(np.arange(n_bins), ymin=np.zeros(n_bins), ymax=o_i[1]),
                ax2.plot(idx1[1], 0, 'ro')[0],))

def compute_kde(dom, w, mu, sigma):
    
    K = norm.pdf(dom, mu, sigma)
    w = w.reshape([w.size, 1])
    
    return np.sum(w*K, axis=0)#/np.sum(w)  

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
from itertools import chain, product
from scipy.stats import norm, rv_discrete
from matplotlib import animation

plt.close('all')

data_eng = es.methods.Data_Engineering()
h5f = data_eng.get_hdf5_file()

# dE = h5f['e_n_HF'][()] - h5f['e_n_LF'][()]
# dZ = h5f['z_n_HF'][()] - h5f['z_n_LF'][()]
# data_eng.set_training_data(feats=['z_n_LF', 'e_n_LF', 'u_n_LF', 's_n_LF',
                                  # 'v_n_LF', 'sprime_n_LF', 'zprime_n_LF'], 
                           # targets = [dE, dZ])

data_eng.set_training_data(feats = ['inner_prods', 'src_Q', 'u_n_LF', 'v_n_LF'],
                           targets = ['dQ'], X_symmetry=[True, False, False, False])
X, y = data_eng.standardize_data()

#idx_triu_0, idx_triu_1 = np.triu_indices(2)
#feat1 = data_eng.h5f['src_Q'][()] 
#feat1 = data_eng.flatten_data([feat1], [False])
#feat2 = data_eng.h5f['inner_prods'][()]
#feat2 = data_eng.flatten_data([feat2], [True])
#feat3 = data_eng.h5f['c_ij'][()]
#feat3 = data_eng.flatten_data([feat3], [False])
#feat4 = data_eng.h5f['Q_HF'][()]
#feat4 = data_eng.flatten_data([feat4], [False])

lags = [[1, 1500]]
X_train, y_train = data_eng.set_lagged_training_data(data_eng.X, lags = lags)

n_train = y_train.shape[0]
n_softmax = y_train.shape[1]
n_bins = 10
n_feat = X_train.shape[1]

data_eng.bin_data(y_train, n_bins)

#mu = []; sigma = []; 
#for i in range(n_softmax):
#    
#    x_p = np.linspace(np.min(y_train[:, i]), np.max(y_train[:, i]), n_bins)
#    #rule of thumb bandwidth selection: https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
#    h = 1.06*np.std(y_train[:, i])*n_samples**(-0.2)
#    sigma_j = np.linspace(1.0*h, 10.0*h, 10)
#    
#    kernel_props = np.array(list(chain(product(x_p, sigma_j))))
#    K = kernel_props.shape[0]
#    mu.append(kernel_props[:, 0].reshape([K,1]))
#    sigma.append(kernel_props[:, 1].reshape([K,1]))
#
train = True
if train:
    ##KVM
    #surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=6, n_neurons=256, 
    #                           n_out=kernel_props.shape[0]*n_softmax, loss='kvm', bias = True,
    #                           activation='relu', activation_out='linear', n_softmax=n_softmax,
    #                           batch_size=512,
    #                           lamb=0.0, decay_step=10**4, decay_rate=0.9, alpha=0.001,
    #                           standardize_X=False, standardize_y=False, save=True,
    #                           kernel_means = mu, kernel_stds = sigma)
    
    #QUANTIZED SOFTMAX
    surrogate = es.methods.ANN(X = X_train, y = data_eng.binned_data, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins*n_softmax, loss = 'cross_entropy', \
                               lamb = 0.0, n_layers = 3, n_neurons=256, activation = 'hard_tanh', activation_out = 'linear', n_softmax = n_softmax, \
                               standardize_X = False, standardize_y = False, batch_size=512, save=True)

    surrogate.train(30000, store_loss = True)
    
    rand_idx = np.random.randint(0, n_train, 5*10**4)
    surrogate.compute_misclass_softmax(X = X_train[rand_idx], y = data_eng.binned_data[rand_idx])
else:
    surrogate = es.methods.ANN(X = np.random.rand(10), y = np.random.rand(10))
    surrogate.load_ANN()
    
make_movie = True
#if True make a movie of the solution, if not just plot final solution
if make_movie:
    
    if surrogate.loss == 'kvm':
    
        n_kde = 100
        n_train = np.int(0.01*X_train.shape[0])
        dom = np.linspace(np.min(y_train, axis=0), np.max(y_train, axis=0), n_kde)
        n_feat = surrogate.n_in
        kde = np.zeros([X_train.shape[0], n_kde, n_softmax])    
        samples = np.zeros([n_train, n_softmax])
        
        for i in range(n_train):
            #w = surrogate.feed_forward(X_train[i].reshape([1, n_feat]))
            w, _, idx = surrogate.get_softmax(X_train[i].reshape([1, n_feat]))
            for j in range(n_softmax):
                kde[i, :, j] = compute_kde(dom[:,j], w[j], mu[j], sigma[j])
                samples[i, j] = norm.rvs(mu[j][idx[j]], sigma[j][idx[j]])
            if np.mod(i, 1000) == 0:
                print('i =', i, 'of', n_train)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim([np.min(dom), np.max(dom)])
        plt.yticks([])
        plt.xlabel(r'SGS data')
           
        plt.tight_layout()
     
        lines = []
        linemarkers = ['-b', '-r']
        symbols = ['bo', 'ro']
        for i in range(2):
            lobj = ax.plot([], [], linemarkers[i])[0]
            lines.append(lobj)
            lobj = ax.plot([], [], symbols[i])[0]
            lines.append(lobj)

    else:
        # n_samples = int(0.001*n_train)
        n_samples = 1000

        ims = []
        fig = plt.figure(figsize=[8,4])
        ax1 = fig.add_subplot(121, xlabel=r'bin number', ylabel=r'probability',
                              title=r'Energy')
        ax2 = fig.add_subplot(122, xlabel=r'bin number', ylabel=r'probability',
                              title='Enstrophy')
        plt.tight_layout()
        
        for i in range(n_samples):
            o_i, _, _ = surrogate.get_softmax(X_train[i].reshape([1, n_feat]))
            idx1 = np.where(data_eng.binned_data[i] == 1.0)[0]
            idx1[1] -= n_bins
            animate()

        im_ani = animation.ArtistAnimation(fig, ims, interval=20, 
                                           repeat_delay=2000, blit=True)