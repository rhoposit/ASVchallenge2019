import numpy as np
import kaldi_io
import sys
import json, h5py

#########################
suffix = str(sys.argv[1])     # '_env_attack', '_env_id', '_attack_id', '' (empty is for speaker)
task = str(sys.argv[2])	      # PA or LA
#########################

def file_len(fname):
   with open(fname) as f:
       for i, l in enumerate(f):
           pass
   return i + 1

workdir = '/disk/scratch1/s1569548/ASVSpoof2019/'

if suffix:
  replacement_key = 'cccCC' 	# use this xvector mean to replace missing xvectors, key from lda_spk_xvector_mat
  replacement_xvectors_path = workdir + 'exp/' + task + '/ASVspoof2019_' + task + '_train/xvectors/xvectors_mfcc_lf0_hf0_env_attack' 
else: 
  replacement_xvectors_path =  workdir + 'exp/' + task + '/ASVspoof2019_' + task + '_train/xvectors/xvectors_mfcc_lf0_hf0' # use mean of all speakers as a replacement

xvec_len = 10

train_xvectors_path = workdir + 'exp/' + task + '/ASVspoof2019_' + task + '_train/xvectors/xvectors_mfcc_lf0_hf0' + suffix
dev_xvectors_path = workdir + 'exp/' + task + '/ASVspoof2019_' + task + '_dev/xvectors/xvectors_mfcc_lf0_hf0' + suffix
eval_xvectors_path = workdir + 'exp/' + task + '/ASVspoof2019_' + task + '_eval/xvectors/xvectors_mfcc_lf0_hf0' + suffix
utt2spk_dev_path = workdir + 'data-xvector' + suffix +'/ASVspoof2019_' + task + '_dev/utt2spk'
utt2spk_eval_path = workdir + 'eval_data' +'/ASVspoof2019_' + task + '_eval/.backup/utt2spk'
utt2spk_train_path = workdir + 'data-xvector' + suffix +'/ASVspoof2019_' + task + '_train/utt2spk'

out_train = train_xvectors_path + '/lda_xvector_mat'
out_dev = dev_xvectors_path + '/lda_xvector_mat'
out_eval = eval_xvectors_path + '/lda_xvector_mat'
out_train_scaled = train_xvectors_path + '/lda_xvector_mat_scaled'
out_dev_scaled = dev_xvectors_path + '/lda_xvector_mat_scaled'
out_eval_scaled = eval_xvectors_path + '/lda_xvector_mat_scaled'

utt2spk_dev = open(utt2spk_dev_path, 'r').readlines()
utt2spk_eval = open(utt2spk_eval_path, 'r').readlines()
utt2spk_train = open(utt2spk_train_path, 'r').readlines()
utts_dev = [ x.strip().split(' ')[0] for x in utt2spk_dev ]
utts_eval = [ x.strip().split(' ')[0] for x in utt2spk_eval ]
utts_train = [x.strip().split(' ')[0] for x in utt2spk_train ]

fea_train = { k:m[0] for k,m in kaldi_io.read_mat_scp(train_xvectors_path+'/lda_xvector_mat.scp') }
fea_dev = { k:m[0] for k,m in kaldi_io.read_mat_scp(dev_xvectors_path+'/lda_xvector_mat.scp') }
fea_eval = { k:m[0] for k,m in kaldi_io.read_mat_scp(eval_xvectors_path+'/lda_xvector_mat.scp') }

print("Missing features number train: ", file_len(utt2spk_train_path) - len(fea_train))
print("Missing features number dev: ", file_len(utt2spk_dev_path) - len(fea_dev))
print("Missing features number eval: ", file_len(utt2spk_eval_path) - len(fea_eval))

if suffix:
  fea_train_spk = { k:m[0] for k,m in kaldi_io.read_mat_scp(replacement_xvectors_path+'/lda_spk_xvector_mat.scp') }
  replacement = fea_train_spk[replacement_key]
  print("Replacing with mean for class", replacement_key)
else:
  replacement = kaldi_io.read_vec_flt(replacement_xvectors_path+'/lda_mean.vec')
  print("Replacing with mean all speaker xvector.")

########## replace missing xvectors ##############

for utt in utts_dev:
    if utt not in fea_dev:
       fea_dev[utt] = replacement 

for utt in utts_eval:
    if utt not in fea_eval:
       fea_eval[utt] = replacement 

for utt in utts_train:
    if utt not in fea_train:
       fea_train[utt] = replacement 

np.save(out_dev, fea_dev)
np.save(out_eval, fea_eval)
np.save(out_train, fea_train)

fd = np.load(out_dev + '.npy')[()]
fe = np.load(out_eval + '.npy')[()]
ft = np.load(out_train + '.npy')[()]

#print(len(fe), list(fe)[0])

print("Missing features number train after replacement: ", file_len(utt2spk_train_path) - len(ft))
print("Missing features number dev after replacement: ", file_len(utt2spk_dev_path) - len(fd))
print("Missing features number eval after replacement: ", file_len(utt2spk_eval_path) - len(fe))

########### Scaled xvectors ############

scaled_fea_dev = { k:0.1*m for k,m in fea_dev.items() }
scaled_fea_eval = { k:0.1*m for k,m in fea_eval.items() }
scaled_fea_train = { k:0.1*m for k,m in fea_train.items() }

np.save(out_dev_scaled, scaled_fea_dev)
np.save(out_eval_scaled, scaled_fea_eval)
np.save(out_train_scaled, scaled_fea_train)

fd = np.load(out_dev_scaled + '.npy')[()]
fe = np.load(out_eval_scaled + '.npy')[()]
ft = np.load(out_train_scaled + '.npy')[()]

sys.exit()

####################################

string_dt = h5py.special_dtype(vlen=str)
D = np.array(list(fea_dev.values()))
T = list(fea_dev.keys())
f = h5py.File(out_dev + '.h5' ,'w')
f.create_dataset("Features", data=D, dtype=np.float32)
f.create_dataset("Targets", data=json.dumps(T), dtype=string_dt)
f.close()
D = np.array(list(fea_train.values()))
T = list(fea_train.keys())
f = h5py.File(out_train + '.h5' ,'w')
f.create_dataset("Features", data=D, dtype=np.float32)
f.create_dataset("Targets", data=json.dumps(T), dtype=string_dt)
f.close()
h5f = h5py.File(out_dev + '.h5' ,'r')
print(h5f['Targets'])
