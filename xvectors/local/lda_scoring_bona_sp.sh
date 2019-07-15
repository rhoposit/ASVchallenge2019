#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains an LDA transform, applies it to the enroll and
# test i-vectors and does cosine scoring.

use_existing_models=false
lda_dim=150
covar_factor=0.0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

lda_data_dir=$1
enroll_data_dir=$2
test_data_dir=$3
lda_ivec_dir=$4
enroll_ivec_dir=$5
test_ivec_dir=$6
trials=$7
scores_dir=$8

if [ "$use_existing_models" == "true" ]; then
  for f in ${lda_ivec_dir}/mean.vec ${lda_ivec_dir}/transform.mat ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
  done
else

mkdir -p ${lda_ivec_dir}/log
run.pl ${lda_ivec_dir}/log/lda.log \
  ivector-compute-lda --dim=$lda_dim  --total-covariance-factor=$covar_factor \
  "ark:ivector-normalize-length scp:${lda_ivec_dir}/xvector.scp ark:- |" \
    ark:${lda_data_dir}/utt2attack \
    ${lda_ivec_dir}/transform.mat || exit 1;
fi

mkdir -p $scores_dir/log
run.pl $scores_dir/log/lda_scoring.log \
  ivector-compute-dot-products  "cat '$trials' | cut -d\  --fields=1,2 |"  \
  "ark:ivector-transform ${lda_ivec_dir}/transform.mat scp:${enroll_ivec_dir}/attack_xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-normalize-length scp:${test_ivec_dir}/xvector.scp ark:- | ivector-transform ${lda_ivec_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  $scores_dir/xvector_mfcc_lf0_hf0_lda_scores || exit 1;
