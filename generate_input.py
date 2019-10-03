"""
GENERATES INPUT JSON FILES FOR REDUCED_SGS.PY
"""

import numpy as np
import json
import os

HOME = os.path.abspath(os.path.dirname(__file__))

#name of the generated input file
input_file = 'EZ_track'

description = "Input file for tracking the value of E and Z"

#target to track ('e', 'z', 'w1', 'w3'), see also get_qoi() subroutine
#in reduced_shs.py
target = ['e', 'z']

#$V_i = \partial q_i/partial \omega$
V = ["-psi_hat_n_LF", "w_hat_n_LF"]

#spectral filter cutoff values per target
k_min = [0, 16]
k_max = [21, 21]

#number of surrogates to be constructed
N_Q = len(target)
fpath = HOME + '/inputs/' + input_file + '.json'

print('Generating input file', fpath)

#remove input file if it already exists
if os.path.isfile(fpath) == True:
    os.remove(fpath)

fp = open(fpath, 'a')

fp.write('%s' %description + '\n')

#simulation flags
flags = {}
flags['sim_ID'] = input_file
flags['state_store'] = False                #store the state at the end of the simulation
flags['restart'] = False                    #restart from previously stored state
flags['store'] = True                       #store data
flags['plot'] = True                        #plot results while running (required drawnow package)
flags['compute_ref'] = True                 #compute the reference solution as well, leave at True, will automatically turn off in surrogate mode
flags['eddy_forcing_type'] = 'tau_ortho'   

json.dump(flags, fp)
fp.write('\n')
fp.write('%d\n' % N_Q)

#write input file
for i in range(N_Q):
    json_in = {}
    json_in['target'] = target[i]
    json_in['V_i'] = V[i]
    json_in['k_min'] = k_min[i]
    json_in['k_max'] = k_max[i]

    json.dump(json_in, fp)
    fp.write('\n')

    print(json_in)
    
fp.close()