#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

#
# Copyright 2019 CSTR, University of Edinburgh
#
# Authors: Erfan Loweimi, Joachim Fainberg, Jennifer Williams, 
#          Joanna Rownicka and Ondrej Klejch
# Apache 2.0.

# modified kaldi/egs/sre16/v2/run.sh

. ./cmd.sh
. ./path.sh

set -e

workdir=/disk/scratch1/s1569548/ASVSpoof2019
mfccdir=$workdir/mfcc
vaddir=$workdir/mfcc
stage=8
nj=16

######## x-vectors ########
task=PA
suffix=_env_id
data_dir=data-xvector$suffix
train_data=ASVspoof2019_${task}_train
dev_data=ASVspoof2019_${task}_dev
dev_data_sps=ASVspoof2019_${task}_dev_sps
xvec_write_dir=exp/$task
xvectors_type=xvectors_mfcc_lf0_hf0$suffix
xvector_stage=4
nnet_dir=$workdir/exp/xvector_nnet_1b${suffix}
num_repeats=200
protocol_dev=$xvec_write_dir/ASVspoof2019_${task}_protocols/ASVspoof2019.${task}.cm.dev.trl.txt
protocol_train=$xvec_write_dir/ASVspoof2019_${task}_protocols/ASVspoof2019.${task}.cm.train.trn.txt

if [ $stage -le 0 ]; then
  # copy features but use spoofing classes as classes for x-vector extraction
  cp -r data-xvector $data_dir
  python local/create${suffix}_files.py $data_dir/$train_data/utt2attack \
    $protocol_train $data_dir/$train_data/utt2"${suffix//_}" || exit 1;
  for dataset in $dev_data $dev_data_sps; do
    python local/create${suffix}_files.py $data_dir/$dataset/utt2attack \
      $protocol_dev $data_dir/$dataset/utt2"${suffix//_}" || exit 1;
  done
  for dataset in $dev_data $dev_data_sps $train_data; do
    mv $data_dir/$dataset/utt2spk $data_dir/$dataset/utt2spk_orig
    mv $data_dir/$dataset/spk2utt $data_dir/$dataset/spk2utt_orig
    mv $data_dir/$dataset/utt2"${suffix//_}" $data_dir/$dataset/utt2spk
    sort -o $data_dir/$dataset/utt2spk $data_dir/$dataset/utt2spk
    utils/utt2spk_to_spk2utt.pl $data_dir/$dataset/utt2spk > $data_dir/$dataset/spk2utt || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  local/nnet3/xvector/run_xvector.sh --stage $xvector_stage --train-stage -1 \
  --num_repeats $num_repeats --num_epochs 3 --data $data_dir/$train_data --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs || exit 1;
fi


if [ $stage -le 2 ]; then
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$train_data \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type || exit 1;
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$dev_data \
    $xvec_write_dir/$dev_data/xvectors/$xvectors_type || exit 1;
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$dev_data_sps \
    $xvec_write_dir/$dev_data_sps/xvectors/$xvectors_type || exit 1;
fi


if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd $xvec_write_dir/$train_data/xvectors/$xvectors_type/log/compute_mean.log \
    ivector-mean scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/xvector.scp \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type/mean.vec || exit 1;
fi

if [ $stage -le 4 ]; then
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd $xvec_write_dir/$train_data/xvectors/$xvectors_type/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/xvector.scp ark:- |" \
    ark:$data_dir/$train_data/utt2spk $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd $xvec_write_dir/$train_data/xvectors/$xvectors_type/log/plda.log \
    ivector-compute-plda ark:$data_dir/$train_data/spk2utt \
    "ark:ivector-subtract-global-mean scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/xvector.scp ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type/plda || exit 1;
fi

if [ $stage -le 5 ]; then
  # create trials file for env and attack classes
  # Trials will look like:
  # aaa00 utt_id target/nontarget
  local/prepare_trials.sh --dir $data_dir/$dev_data --trials-suffix $suffix || exit 1;
fi

if [ $stage -le 6 ]; then
  trials=$data_dir/$dev_data/trials$suffix
  cat $trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
    scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/spk_xvector.scp \
    "ark:ivector-normalize-length scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/xvector.scp ark:- |" \
    exp/scores/${xvectors_type}_cos_scores
  eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/${xvectors_type}_cos_scores) 2> /dev/null`
  echo "$eer" | tee -a exp/scores/${xvectors_type}_cos_scores_eer
fi

if [ $stage -le 7 ]; then
  # do the PLDA scoring. Use training set as the enrollment set.
  trials=$data_dir/$dev_data/trials$suffix
  $train_cmd exp/scores/log/${xvectors_type}_plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$xvec_write_dir/$train_data/xvectors/$xvectors_type/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $xvec_write_dir/$train_data/xvectors/$xvectors_type/plda - |" \
    "ark:ivector-mean ark:$data_dir/$train_data/spk2utt scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/xvector.scp ark:- | ivector-subtract-global-mean $xvec_write_dir/$train_data/xvectors/$xvectors_type/mean.vec ark:- ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $xvec_write_dir/$train_data/xvectors/$xvectors_type/mean.vec scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/xvector.scp ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" exp/scores/${xvectors_type}_plda_scores || exit 1;
  eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/${xvectors_type}_plda_scores) 2> /dev/null`
  echo "$eer" | tee -a exp/scores/${xvectors_type}_plda_scores_eer
fi

if [ $stage -le 8 ]; then
  trials=$data_dir/$dev_data/trials$suffix
  lda_dim=10
  local/lda_scoring.sh --lda-dim $lda_dim $data_dir/$train_data $data_dir/$train_data \
    $data_dir/$dev_data $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$dev_data/xvectors/$xvectors_type \
    $trials exp/scores $xvectors_type || exit 1;
    eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/${xvectors_type}_lda_scores) 2> /dev/null`
    echo "LDA${lda_dim}: $eer" | tee -a exp/scores/${xvectors_type}_lda_scores_eer
fi

if [ $stage -le 9 ]; then

  ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/xvector.scp ark:- |\
    ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_xvector.ark,$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_xvector.scp || exit 1;
  sed -e 's@\[ @\[ \n@g' <(copy-vector scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_xvector.scp ark,t:-) |\
     copy-matrix ark,t:- ark,scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_xvector_mat.ark,$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_xvector_mat.scp

  ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/spk_xvector.scp ark:- |\
    ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_spk_xvector.ark,$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_spk_xvector.scp || exit 1;
  sed -e 's@\[ @\[ \n@g' <(copy-vector scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_spk_xvector.scp ark,t:-) |\
     copy-matrix ark,t:- ark,scp:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_spk_xvector_mat.ark,$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_spk_xvector_mat.scp

  ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/xvector.scp ark:- |\
    ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector.ark,$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector.scp || exit 1;
   sed -e 's@\[ @\[ \n@g' <(copy-vector scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector.scp ark,t:-) |\
     copy-matrix ark,t:- ark,scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector_mat.ark,$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector_mat.scp
fi

exit

#TODO
# mapping from env_attack classes to bonafide/spoofed
# 

# create CM score file
# take kaldi scores, convert to log-probability (with sigmoid), and compute llk ratio 
if [ $stage -le 10 ]; then
  file_with_scores=exp/scores/${xvectors_type}_lda_scores
  python local/create_cm_score_file.py $data_dir/$dev_data/utt2attack $file_with_scores \
    $protocol_dev ${file_with_scores}_cm || exit 1;
fi

if [ $stage -le 11 ]; then
  # get the tDCF score
  file_with_scores=exp/scores/${xvectors_type}_plda_scores
  echo $file_with_scores > ${file_with_scores}_tdcf
  python $workdir/asv/tDCF_python_v1/evaluate_tDCF_asvspoof19.py \
    $workdir/${file_with_scores}_cm $xvec_write_dir/ASVspoof2019_PA_dev_asv_scores_v1.txt |\
     tee -a $workdir/${file_with_scores}_tdcf || exit 1;
fi
