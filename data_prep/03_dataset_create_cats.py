import numpy as np
import sklearn.preprocessing
from collections import defaultdict
from operator import itemgetter
import os, sys, glob, json
import h5py
import librosa
import librosa.display
from scipy import signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################################################################
# To run:
# python 03_dataset_create.py [train|dev|eval] [feat]
#
# Example:
# python 03_dataset_create.py train mfccDD 
# python 03_dataset_create.py dev mfccDD
# feats_types = ["mfccDD", "imfccDD", "rfccDD", "scmcDD"] etc
###############################################################

# command line args
dataset = sys.argv[1]
feat = sys.argv[2]
task = "PA"

########################################################################


def resample(y, n):
    x = np.arange(len(y))
    y_new = signal.resample(y, n)
    return y_new


def load_csv(infile):
    input = open(infile, "r")
    data = input.read().split("\n")[:-1]
    input.close()
    truth = defaultdict(list)
    truth_attack = defaultdict(list)
    for item in data:
        fname, envID, attackID, t = item.split(",")
        fname = fname.split(".wav")[0]
        truth[fname] = item
        truth_attack[envID].append(item)
    return truth, truth_attack


def get_fnames_subset(attack, truth):
    new_target_subsets = []
    items = truth[attack]
    for entry in items:
        fname = entry.split(",")[0].split(".wav")[0]
        new_target_subsets.append(fname)
    items = truth["none"]
    for entry in items:
        fname = entry.split(",")[0].split(".wav")[0]
        new_target_subsets.append(fname)
    return new_target_subsets



########################################################################

count = 0
#filelist = glob.glob(task+"_feats_"+dataset+"/"+feat+"/*.h5")
coefs_ref = {"mfcc":70, "imfcc":60, "rfcc":30, "scmc":40}
num_coefs = coefs_ref[feat]

truth, truth_attack = load_csv(task+"_reference_"+dataset+".csv")
framez = []

n = 10

if task == "PA":
#    attack_types = ["AA", "AB", "AC", "BA", "BB", "BC", "CA", "CB", "CC"]
    attack_types = ["aaa", "aab", "aac", "aba", "abb", "abc", "aca", "acb", "acc", "bbb", "bba", "bbc", "baa", "bac", "bab", "bca", "bcb", "bcc", "ccc", "cca", "ccb", "caa", "cab", "cac", "cba", "cbc", "cbb"]

    
if task == "LA":
    attack_types = ["SS_1", "SS_2", "SS_4", "US_1", "VC_1", "VC_4"]

    
for attack in attack_types:
    new_target_subsets = get_fnames_subset(attack, truth_attack)
    total = len(new_target_subsets)
    T = []
    NEW = []
    count = 0
    for f in new_target_subsets:
        f = task+"_feats_"+dataset+"/"+feat+"/"+f+".h5"
        h5f = h5py.File(f,'r')
        dat = h5f[feat+"DD"][:]
        h5f.close()
        count += 1
        print("file "+str(count)+" of "+str(total))
        num_frames = np.array(dat[:,:num_coefs]).shape[0]
        framez.append(num_frames)
        fkey = f.split(".h5")[0].split("/")[-1]
        s = dat[:,:num_coefs]
        d = (dat[:,num_coefs:(num_coefs*2)])
        dd = dat[:,(num_coefs*2):]
        t = truth[fkey]
        s_new = resample(s, n) # frames(rows) x coefs(columns)
        s_new = s_new.flatten() # 1 x (coefs*frames): [coefs_f1][coefs_f2][coefs_f3]..
        d_new = resample(d, n).flatten()
        dd_new = resample(dd, n).flatten()
        interm = np.concatenate((s_new, d_new), axis=None)
        final = np.concatenate((interm, dd_new), axis=None) # mfcc, mfccD, mfccDD
        NEW.append(final)
        # need ground truth to go with this stacked representation
        T.append(t)


    
    D = np.array(NEW)
    outfile = task+"_"+dataset+"_resample/"+task+"_"+attack+"_"+dataset+"_"+feat+"_bin"+str(n)+".h5"
    f = h5py.File(outfile,'w')
    dset = f.create_dataset("Features", data=D, dtype=np.float)
    dset = f.create_dataset("Targets", data=json.dumps(T))
    f.close()
