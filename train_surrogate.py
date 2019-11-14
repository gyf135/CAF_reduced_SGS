import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import tkinter as tk
from tkinter import filedialog
import h5py
from itertools import chain, product

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

h5f = h5py.File(file_path, 'r')

feats = ['e_n_LF', 'c_12', 'c_22']
targets = ['e_n_HF'] 

n_samples = h5f[feats[0]][:].size

X = np.zeros([n_samples, len(feats)])
y = np.zeros([n_samples, len(targets)])

for i in range(len(feats)):
    X[:, i] = h5f[feats[i]][:]
    
for i in range(len(targets)):
    y[:, i] = h5f[targets[i]]

feat_eng = es.methods.Feature_Engineering(X, y)
X, y = feat_eng.standardize_data()
lags = [[20, 30]]
X_train, y_train = feat_eng.lag_training_data(X, lags = lags)

n_softmax = 0#len(targets)
n_bins = 20

x_p = np.linspace(np.min(y_train), np.max(y_train), n_bins)

#rule of thumb bandwidth selection: https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
h = 1.06*np.std(y_train)*n_samples**(-0.2)
sigma_j = np.linspace(1.0*h, 1.0*h, 1)

kernel_props = np.array(list(chain(product(x_p, sigma_j))))

mu = kernel_props[:, 0]
sigma = kernel_props[:, 1]
mu = mu.reshape([mu.size, 1])
sigma = sigma.reshape([sigma.size, 1])

surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=256, 
                           n_out=1, loss='squared', bias = True,
                           activation='tanh', activation_out='linear', n_softmax=n_softmax,
                           batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, alpha=0.0001,
                           standardize_X=False, standardize_y=False, save=False,
                           kernel_means = mu, kernel_stds = sigma)
surrogate.train(30000, store_loss = True)

