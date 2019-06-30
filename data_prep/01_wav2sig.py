import numpy as np
import sklearn.preprocessing
from collections import defaultdict
from operator import itemgetter
import sys, glob
import librosa
import librosa.display
import h5py


####################################################################
# To Run:
# python 01_wav2sig /path/to/wav/ /path/to/h5/folder
####################################################################

wavs = sys.argv[1]
sig_folder = sys.argv[2]

filelist = glob.glob(wavs+"/*.wav")
count = 0
skipped = 0
total = len(filelist)

for w in filelist:
    count += 1
    print(w)
    print(str(count)+", "+str(total))
    y, sr = librosa.load(w, sr=16000)
    new_file = sig_folder+"/"+w.split("/")[-1].split(".wav")[0]+".h5"
    h5f = h5py.File(new_file, 'w')
    h5f.create_dataset('dataset', data=y, chunks=True)
    print(new_file)
    h5f.close

