asv_dir=/group/project/cstr1/asv2019/eval/
dataset_name=ASVspoof2019_PA_eval
kaldi_data_dir=/disk/scratch1/s1569548/ASVSpoof2019/eval_data/$dataset_name

mkdir -p $kaldi_data_dir

ls $asv_dir/$dataset_name/flac | sed 's@.flac@@' > $kaldi_data_dir/utt_list
awk '{print $1, $1}' $kaldi_data_dir/utt_list > $kaldi_data_dir/utt2spk
cp $kaldi_data_dir/utt2spk $kaldi_data_dir/spk2utt

wav_file=$kaldi_data_dir/wav_list
wav_scp=$kaldi_data_dir/wav.scp

if [[ ! -f $wav_scp ]]; then
  echo -e "\n $0 [info]: Start creating wav.scp..."
  sed    s:^:"sox -t flac $asv_dir/$dataset_name/flac/":g $kaldi_data_dir/utt_list > $wav_file
  sed -i s:$:".flac -t wav - |":g $wav_file
  paste -d' ' $kaldi_data_dir/utt_list $wav_file > $wav_scp
fi
