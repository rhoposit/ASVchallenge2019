import os
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import defaultdict
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, sys
from pathlib import Path

####################################################################
# To Run:
# python 00_flac2wav.py protocols_file /path/to/flac/ /path/to/wav/
####################################################################

protocols_file = sys.argv[1]
flac_path = sys.argv[2]
wav_dir = sys.argv[3]


flac_files = glob.glob(flac_path+"/*.flac")
input = open(protocols_file, "r")
protocols = input.read().split("\n")[:-1]
input.close()

for item in protocols:
    new_filename = wav_dir+"/"+item+".wav"
    if not Path(new_filename).is_file():
        command = "ffmpeg -i "+flac_path+"/"+item+".flac "+new_filename
        print(command)
        os.system(command)
