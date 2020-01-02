import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import h5py
import tkinter as tk

root = tk.Tk()
root.withdraw()
fname = tk.filedialog.askopenfilename()
h5f = h5py.File(fname, 'r')
print(h5f.keys())

root = tk.Tk()
root.withdraw()
fname_ref = tk.filedialog.askopenfilename()
h5f_ref = h5py.File(fname_ref, 'r')
print(h5f.keys())

post_proc = es.methods.Post_Processing()

N_Q = h5f['Q_LF'][()].shape[1]

fig = plt.figure(figsize=[N_Q*4, 4])

sub = 10

for q in range(N_Q):

    ax = fig.add_subplot(1, N_Q, q+1)
    
    Q_HF_q = h5f_ref['Q_HF'][()][0:-1:sub, q]
    Q_LF_q = h5f['Q_LF'][()][0:-1:sub, q]
    
    dom_LF, kde_LF = post_proc.get_pde(Q_LF_q)
    dom_HF, kde_HF = post_proc.get_pde(Q_HF_q)
    
    ax.plot(dom_LF, kde_LF)
    ax.plot(dom_HF, kde_HF, '--k')
   
h5f.close()
plt.show()