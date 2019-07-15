#!/bin/bash

# Joanna Rownicka
# The University of Edinburgh

# generate trials file needed for scoring

dir=data-xvector_env_attack/ASVspoof2019_PA_dev
trials_suffix=_env_attack

# percentage of nontarget trials
nontarget_proportion=50

. utils/parse_options.sh

# target part of trials
cat $dir/utt2spk | awk '{print $2,$1,"target"}' > $dir/target${trials_suffix}

# calcuate number of nontarget trials to generate given the desired proportion
n_utt_target=`wc -l <$dir/utt2spk`
n_utt_nontarget=$(( $nontarget_proportion*$n_utt_target / (100 - $nontarget_proportion) ))

if [ -f $dir/nontarget$trials_suffix ]; then
  rm $dir/nontarget$trials_suffix || exit 1;
fi

#nontarget part of trials
for i in $(seq 1 $n_utt_nontarget); do
  # random speaker
  shuf -n 1 $dir/spk2utt | awk '{printf ("%s ", $1)}' > $dir/rand_nontarget$trials_suffix
  # random mismatched utterance (from a different speaker)
  cat $dir/rand_nontarget$trials_suffix | awk '{print $1}' | \
    xargs -I {} awk '$2 !~ /{}/ {print $0}' $dir/utt2spk | \
    shuf -n 1 | awk '{printf $1}' >> $dir/rand_nontarget$trials_suffix
  awk '{print $0,"nontarget"}' $dir/rand_nontarget$trials_suffix >> $dir/nontarget$trials_suffix
done

cat $dir/target$trials_suffix $dir/nontarget$trials_suffix | sort > $dir/trials${trials_suffix}
# cleanup
rm $dir/target$trials_suffix $dir/nontarget$trials_suffix $dir/rand_nontarget$trials_suffix 
