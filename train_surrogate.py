def animate(s):

    I = 2
    for i in range(I):
        lines[2*i].set_data(dom[:, i], kde[s, :, i]/np.max(kde[s, :, i]))
        lines[2*i + 1].set_data(y_train[s, i], 0.0)

    return lines

def compute_kde(dom, w, mu, sigma):
    
    K = norm.pdf(dom, mu, sigma)
    w = w.reshape([w.size, 1])
    
    return np.sum(w*K, axis=0)#/np.sum(w)  

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
from itertools import chain, product
from scipy.stats import norm, rv_discrete
from matplotlib.animation import FuncAnimation, writers

feat_eng = es.methods.Feature_Engineering(feat_names=['inner_prods'], 
                                          target = 'dQ',
                                          X_symmetry = [True])
X, y = feat_eng.standardize_data()
lags = [[1, 30]]
X_train, y_train = feat_eng.lag_training_data(feat_eng.X, lags = lags)

n_samples = y_train.shape[0]
n_softmax = y_train.shape[1]
n_bins = 20

mu = []; sigma = []; 
for i in range(n_softmax):
    
    x_p = np.linspace(np.min(y_train[:, i]), np.max(y_train[:, i]), n_bins)
    #rule of thumb bandwidth selection: https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
    h = 1.06*np.std(y_train[:, i])*n_samples**(-0.2)
    sigma_j = np.linspace(1.0*h, 10.0*h, 10)
    
    kernel_props = np.array(list(chain(product(x_p, sigma_j))))
    K = kernel_props.shape[0]
    mu.append(kernel_props[:, 0].reshape([K,1]))
    sigma.append(kernel_props[:, 1].reshape([K,1]))

#kernel_props = np.concatenate(kernel_props, axis=0)

surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=256, 
                           n_out=kernel_props.shape[0]*n_softmax, loss='kvm', bias = True,
                           activation='hard_tanh', activation_out='linear', n_softmax=n_softmax,
                           batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, alpha=0.001,
                           standardize_X=False, standardize_y=False, save=True,
                           kernel_means = mu, kernel_stds = sigma)
surrogate.train(10000, store_loss = True)

make_movie = True
#if True make a movie of the solution, if not just plot final solution
if make_movie:
    
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
    plt.xlabel(r'$y$')
       
    plt.tight_layout()
 
    lines = []
    linemarkers = ['-b', '-r']
    symbols = ['bo', 'ro']
    for i in range(2):
        lobj = ax.plot([], [], linemarkers[i])[0]
        lines.append(lobj)
        lobj = ax.plot([], [], symbols[i])[0]
        lines.append(lobj)
    
    # Set up formatting for the movie files
    anim = FuncAnimation(fig, animate, frames=np.arange(0, n_train, 100))    
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    anim.save('demo.mp4', writer = writer)