#!/bin/bash

CUDA_VISIBLE_DEVICES=0

. ./cmd.sh
. ./path.sh

set -e

workdir=/disk/scratch1/s1569548/ASVSpoof2019
mfccdir=$workdir/mfcc
vaddir=$workdir/mfcc
stage=0
nj=16

task=PA
data_dir=eval_data
train_data=ASVspoof2019_${task}_train
eval_data=ASVspoof2019_${task}_eval
xvec_write_dir=exp/$task

for suffix in _env_attack _env_id _attack_id ''; do

  if  [ -z "$suffix" ];
  then 
    nnet_dir=$workdir/exp/xvector_nnet_1a
    echo "Extracting xvectors for speakers."
  else
    nnet_dir=$workdir/exp/xvector_nnet_1b${suffix}
    echo "Extracting xvectors for $suffix"
  fi

  if [ $stage -le 0 ]; then
    local/prepare_eval_data.sh || exit 1;
  fi
  
  if [ $stage -le 1 ]; then
    local/asv_kaldi_mfcc_ext.sh --write-utt2num-frames true \
      --feat-type mfcc-xvector --kaldi-data-dir $data_dir || exit 1;
  fi
  
  if [ $stage -le 2 ]; then
    local/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      eval_data/$eval_data exp/make_vad $vaddir
    utils/fix_data_dir.sh eval_data/$eval_data || exit 1;
  fi
  
  if [ $stage -le 3 ]; then
    xvectors_type=xvectors_mfcc_lf0_hf0$suffix
    local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
      $nnet_dir $data_dir/$eval_data \
      $xvec_write_dir/$eval_data/xvectors/$xvectors_type || exit 1;
  fi
  
  # 
  if [ $stage -le 3 ]; then
    xvectors_type=xvectors_mfcc_lf0_hf0$suffix
    ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat scp:$xvec_write_dir/$eval_data/xvectors/$xvectors_type/xvector.scp ark:- |\
      ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$eval_data/xvectors/$xvectors_type/lda_xvector.ark,$xvec_write_dir/$eval_data/xvectors/$xvectors_type/lda_xvector.scp || exit 1;
     sed -e 's@\[ @\[ \n@g' <(copy-vector scp:$xvec_write_dir/$eval_data/xvectors/$xvectors_type/lda_xvector.scp ark,t:-) |\
       copy-matrix ark,t:- ark,scp:$xvec_write_dir/$eval_data/xvectors/$xvectors_type/lda_xvector_mat.ark,$xvec_write_dir/$eval_data/xvectors/$xvectors_type/lda_xvector_mat.scp
  fi
 
  # converts all sets (train, dev, eval), todo change to specified set 
  if [ $stage -le 4 ]; then
    python local/convert_to_numpy.py "$suffix" "$task" || exit 1;
  fi
