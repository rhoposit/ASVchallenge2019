#!/bin/bash

CUDA_VISIBLE_DEVICES=0

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
stage=14
nj=16

######## x-vectors ########
task=PA
suffix=
data_dir=data-xvector
train_data=ASVspoof2019_${task}_train
dev_data=ASVspoof2019_${task}_dev
dev_data_sps=ASVspoof2019_${task}_dev_sps
xvec_write_dir=exp/$task
#xvec_write_dir=/group/project/cstr1/asv2019/train_dev/$task
xvectors_type=xvectors_mfcc_lf0_hf0$suffix
xvector_stage=6
nnet_dir=$workdir/exp/xvector_nnet_1a
protocol_dev=$xvec_write_dir/ASVspoof2019_${task}_protocols/ASVspoof2019.${task}.cm.dev.trl.txt
protocol_train=$xvec_write_dir/ASVspoof2019_${task}_protocols/ASVspoof2019.${task}.cm.train.trn.txt

if [ $stage -le 0 ]; then
  cp -r data $data_dir
  ./asv/kaldi/asv_kaldi_mfcc_ext.sh --write-utt2num-frames true \
    --feat-type mfcc-xvector --kaldi-data-dir $data_dir || exit 1;
fi

# prepare dev data dir such that each utt is mapped to a new speaker
if [ $stage -le 1 ]; then
  mkdir $data_dir/$dev_data_sps
  cp $data_dir/$dev_data/{feats.scp,utt2spk,vad.scp,wav.scp,utt2num_frames,utt2bon,utt2spf} $data_dir/$dev_data_sps
  awk '{print $1, $1}' $data_dir/$dev_data_sps/utt2spk > $data_dir/$dev_data_sps/utt2spk_new
  rm $data_dir/$dev_data_sps/utt2spk; mv $data_dir/$dev_data_sps/utt2spk_new $data_dir/$dev_data_sps/utt2spk;
  utils/utt2spk_to_spk2utt.pl $data_dir/$dev_data_sps/utt2spk > $data_dir/$dev_data_sps/spk2utt
  utils/validate_data_dir.sh --no-text $data_dir/$dev_data_sps || exit 1;
fi

if [ $stage -le 2 ]; then
  local/nnet3/xvector/run_xvector.sh --stage $xvector_stage --train-stage -1 \
  --data $data_dir/$train_data --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs || exit 1;
fi

if [ $stage -le 3 ]; then
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$train_data \
    $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0 || exit 1;
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$dev_data \
    $xvec_write_dir/$dev_data/xvectors/xvectors_mfcc_lf0_hf0 || exit 1;
  local/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj $nj \
    $nnet_dir $data_dir/$dev_data_sps \
    $xvec_write_dir/$dev_data_sps/xvectors/xvectors_mfcc_lf0_hf0 || exit 1;
fi

if [ $stage -le 4 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/log/compute_mean.log \
    ivector-mean scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp \
    $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/mean.vec || exit 1;
fi

if [ $stage -le 5 ]; then
  # create reference files, for binafide/spoofed classification
  for dataset in $train_data $dev_data $dev_data_sps; do
    cat $data_dir/$dataset/utt2bon $data_dir/$dataset/utt2spf | sort > $data_dir/$dataset/utt2attack
    utils/utt2spk_to_spk2utt.pl $data_dir/$dataset/utt2attack > $data_dir/$dataset/attack2utt || exit 1;
  done

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp ark:- |" \
    ark:$data_dir/$train_data/utt2attack $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/log/plda.log \
    ivector-compute-plda ark:$data_dir/$train_data/attack2utt \
    "ark:ivector-subtract-global-mean scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/plda || exit 1;
fi

if [ $stage -le 6 ]; then
  # Compute attack-level i-vectors and num_utts per condition. This will be needed for scoring.
  # Be careful here: the speaker-level iVectors are now length-normalized,
  # even if they are otherwise the same as the utterance-level ones.
  echo "$0: computing mean of iVectors for spoofed/bonafide classes and length-normalizing"
  run.pl $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/log/attack_mean.log \
    ivector-normalize-length scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp  ark:- \| \
    ivector-mean ark:$data_dir/$train_data/attack2utt ark:- ark:- ark,t:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/num_utts_attack.ark \| \
    ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/attack_ivector.ark,$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/attack_xvector.scp || exit 1;
fi

if [ $stage -le 7 ]; then
  # create trials file
  # "high detection score should indicate bona fide and low score should indicate spoofing attack"
  # We'll evaluate each file as a bonafide trial and as a spoofed trial to get the llk ratio
  # Trials will look like:
  # bonafide utt_id target
  # spoof utt_id nontarget
  awk '{
    if ($2=="bonafide") print "bonafide", $1, "target";
    else print "bonafide", $1, "nontarget"
    }' $data_dir/$dev_data/utt2attack | sort > $data_dir/$dev_data/trials
  awk '{
    if ($2=="spoof") print "spoof", $1, "target";
    else print "spoof", $1, "nontarget"
    }' $data_dir/$dev_data/utt2attack | sort >> $data_dir/$dev_data/trials
fi

if [ $stage -le 8 ]; then
  trials=$data_dir/$dev_data/trials
  cat $trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
    scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/attack_xvector.scp \
    "ark:ivector-normalize-length scp:$xvec_write_dir/$dev_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp ark:- |" \
    exp/scores/xvector_mfcc_lf0_hf0_cos_scores
  eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/xvector_mfcc_lf0_hf0_cos_scores) 2> /dev/null`
  echo "$eer" | tee -a exp/scores/xvector_mfcc_lf0_hf0_cos_scores_eer
fi

if [ $stage -le 9 ]; then
  # do the PLDA scoring. Use training set as the enrollment set.
  $train_cmd exp/scores/log/xvector_mfcc_lf0_hf0_plda_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/num_utts_attack.ark \
    "ivector-copy-plda --smoothing=0.0 $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/plda - |" \
    "ark:ivector-mean ark:$data_dir/$train_data/attack2utt scp:$xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp ark:- | ivector-subtract-global-mean $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/mean.vec ark:- ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/mean.vec scp:$xvec_write_dir/$dev_data/xvectors/xvectors_mfcc_lf0_hf0/xvector.scp ark:- | transform-vec $xvec_write_dir/$train_data/xvectors/xvectors_mfcc_lf0_hf0/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data_dir/$dev_data/trials' | cut -d\  --fields=1,2 |" exp/scores/xvector_mfcc_lf0_hf0_plda_scores || exit 1;
  eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/xvector_mfcc_lf0_hf0_plda_scores) 2> /dev/null`
  echo "$eer" | tee -a exp/scores/xvector_mfcc_lf0_hf0_plda_scores_eer
fi


if [ $stage -le 10 ]; then
  trials=$data_dir/$dev_data/trials
  lda_dim=500
  xvectors_type=xvectors_mfcc_lf0_hf0
  local/lda_scoring_bona_sp.sh --lda-dim $lda_dim $data_dir/$train_data $data_dir/$train_data \
    $data_dir/$dev_data $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$dev_data/xvectors/$xvectors_type \
    $trials exp/scores || exit 1;
    eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/${xvectors_type}_lda_scores) 2> /dev/null`
    echo "LDA${lda_dim}: $eer" | tee -a exp/scores/${xvectors_type}_lda_scores_eer

  ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/xvector.scp ark:- |\
    ivector-normalize-length ark:- ark,scp:$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector.ark,$xvec_write_dir/$dev_data/xvectors/$xvectors_type/lda_xvector.scp || exit 1;

fi

# create CM score file
# take kaldi scores, convert to log-probability (with sigmoid), and compute llk ratio 
if [ $stage -le 11 ]; then
  file_with_scores=exp/scores/xvector_mfcc_lf0_hf0_plda_scores
  python local/create_cm_score_file.py $data_dir/$dev_data/utt2attack $file_with_scores \
    $xvec_write_dir/ASVspoof2019_${task}_protocols/ASVspoof2019.${task}.cm.dev.trl.txt \
    ${file_with_scores}_cm || exit 1;
fi

if [ $stage -le 12 ]; then
  # get the tDCF score
  file_with_scores=exp/scores/xvector_mfcc_lf0_hf0_plda_scores
  echo $file_with_scores > ${file_with_scores}_tdcf
  python $workdir/asv/tDCF_python_v1/evaluate_tDCF_asvspoof19.py \
    $workdir/${file_with_scores}_cm $xvec_write_dir/ASVspoof2019_PA_dev_asv_scores_v1.txt |\
     tee -a $workdir/${file_with_scores}_tdcf || exit 1;
fi

if [ $stage -le 13 ]; then
  # create trials file for env and attack classes
  # Trials will look like:
  # spk_id utt_id target
  # nontarget proportion = 0
  local/prepare_trials_train.sh --dir $data_dir/$train_data --trials-suffix _train_spk || exit 1;
fi

# those steps are just for the lda transform, scoring not reliable (on train data only)

if [ $stage -le 14 ]; then
  trials=$data_dir/$train_data/trials_train_spk
  lda_dim=10
  local/lda_scoring.sh --lda-dim $lda_dim $data_dir/$train_data $data_dir/$train_data \
    $data_dir/$train_data $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $xvec_write_dir/$train_data/xvectors/$xvectors_type \
    $trials exp/scores $xvectors_type || exit 1;
    eer=`compute-eer <(python local/prepare_for_eer.py $trials exp/scores/${xvectors_type}_lda_scores)`
    echo "LDA${lda_dim}: $eer" | tee -a exp/scores/${xvectors_type}_lda_scores_eer
fi


if [ $stage -le 15 ]; then

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

  ivector-transform $xvec_write_dir/$train_data/xvectors/$xvectors_type/transform.mat ark,t:$xvec_write_dir/$train_data/xvectors/$xvectors_type/mean.ark ark:- |\
    ivector-normalize-length ark:- ark,t:$xvec_write_dir/$train_data/xvectors/$xvectors_type/lda_mean.ark || exit 1;
fi

