#!/bin/bash

#
# Copyright 2019 CSTR, University of Edinburgh
#
# Authors: Erfan Loweimi, Joachim Fainberg, Jennifer Williams, 
#          Joanna Rownicka and Ondrej Klejch
# Apache 2.0.

# Description:
# This script extracts mfcc for ASV2019
# 

# Path to Kaldi s5 dir ...
kaldi_asv_s5_dir=/disk/scratch1/s1569548/ASVSpoof2019   # <<< SET THIS PATH <<<
kaldi_steps_dir=$kaldi_asv_s5_dir/steps                 # <<< SET THIS PATH <<<
kaldi_data_dir=$kaldi_asv_s5_dir/data                   # <<< SET THIS PATH <<<
feat_type=mfcc_hres


# Config here ...
feat_config=$kaldi_asv_s5_dir/conf/mfcc_hres.conf
nj=32
write_utt2num_frames=false

. parse_options.sh || exit 1;
feat_dir=$kaldi_asv_s5_dir/${feat_type}

#datasets=('ASVspoof2019_PA_train' 'ASVspoof2019_PA_dev'
#          'ASVspoof2019_LA_train' 'ASVspoof2019_LA_dev')

datasets=('ASVspoof2019_PA_train' 'ASVspoof2019_PA_dev')

#datasets=('ASVspoof2019_PA_eval')

# MFCC Feature Extraction ...
mkdir -p $feat_dir
for dataset in ${datasets[@]}; do 
    $kaldi_steps_dir/make_mfcc.sh --write-utt2num-frames $write_utt2num_frames \
        --mfcc-config $feat_config --nj $nj $kaldi_data_dir/$dataset \
        exp/make_$feat_type/$dataset $feat_dir || exit 1;
    #$kaldi_steps_dir/compute_cmvn_stats.sh $kaldi_data_dir/$dataset || exit 1;
done
