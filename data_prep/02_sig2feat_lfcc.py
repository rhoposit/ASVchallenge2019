import numpy as np
import sklearn.preprocessing
from collections import defaultdict
from operator import itemgetter
import sys, glob
import h5py
import librosa
import librosa.display

import bob
import bob.ap


####################################################################
# To Run:
# python 02_sig2feat.py [train|dev|eval]
####################################################################

dataset = sys.argv[1]
task = "PA"



def make_feats(y, sr, feats_dir, f):
    win_length_ms = 20 # The window length of the cepstral analysis in milliseconds
    win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
    n_filters = 24 # The number of filter bands
    n_ceps = 19 # The number of cepstral coefficients
    delta_win = 2 # The integer delta value used for computing the first and second order derivatives
    dct_norm = True # A factor by which the cepstral coefficients are multiplied
    pre_emphasis_coef = 0.97
    
    # ** make MFCCs
    # Mel Filterbannk Cepstral Coefficients
    mel_scale = False
    n_ceps = 70
    f_min = 100. # The minimal frequency of the filter bank
    f_max = 7800. # The maximal frequency of the filter bank
    c = bob.ap.Ceps(sr, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    c.f_min = 300. # The minimal frequency of the filter bank
    c.f_max = 8000. # The maximal frequency of the filter bank
    signal = np.cast['float'](y)
    # ** make LFCCs - delta delta
    c.with_delta_delta = True
    lfccDD = c(signal)
    fname = feats_dir+"/lfcc/"+f
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('lfcc', data=lfccDD)
    h5f.close()


    
types_ref = ["lfcc", "lfccD", "lfccDD"]



count = 0
filelist = glob.glob(task+"_h5py_"+dataset+"/*.h5")
total = len(filelist)
for f in filelist:
    h5f = h5py.File(f,'r')
    y = h5f['dataset'][:]
    h5f.close()
    feats_dir = task+"_feats_"+dataset+"/"
    fname = f.split("/")[-1]
    sr = 16000
    make_feats(y, sr, feats_dir, fname)
    count += 1
    print(f)
    print(str(count)+", "+str(total))
