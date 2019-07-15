#
# Copyright 2019 CSTR, University of Edinburgh
#
# Authors: Erfan Loweimi, Joachim Fainberg, Jennifer Williams, 
#          Joanna Rownicka and Ondrej Klejch
# Apache 2.0.

# Given a gold label file (utt2attack) and scores produced by kaldi scoring (kaldi_scores),
# compute llk ratio and create scores_cm file for tDCF scoring

import sys
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

utt2attack = open(sys.argv[1], 'r').readlines()
kaldi_scores = open(sys.argv[2], 'r').readlines()
protocol = open(sys.argv[3], 'r').readlines()
cm_scores_file = open(sys.argv[4], 'w+')

hash_table = dict()
hash_table = { x.split(' ')[0].strip(): [x.split(' ')[1].strip()] for x in utt2attack }
protocol_table = [ x.strip().split(' ') for x in protocol ]

for line in kaldi_scores:
    # e.g. 'PA_0074-D_0013837': ['spoof', ['bonafide', '-3.614832'], ['spoof', '-0.3100256']]
    l = line.strip().split(' ')
    test_case = l[0]
    utt_id = l[1]
    score = l[2]
    hash_table[utt_id].append([test_case, score])

n = 0
for k,v in hash_table.items():
    utt_id = k
    true_label = v[0]
    if len(v) == 1:
        n += 1
        print "No score for file ", utt_id, " - removing from scoring."
        hash_table.pop(utt_id)
        continue
    if v[1][0] == 'bonafide':
        idxs = (1,2)
    else:
        idxs = (2,1)
    
    # convert to probability and compute log-likelihood ratio
    llk_genuine = math.log(sigmoid(float(v[idxs[0]][1])))
    llk_spoof = math.log(sigmoid(float(v[idxs[1]][1])))
    cm_score = llk_genuine - llk_spoof

    # look up the attack type in the protocol
    for item in protocol_table:
        if item[1] == utt_id[0:3]+utt_id[8:]:
            attack_type = item[3]
            break

    cm_scores_file.write(utt_id[0:3]+utt_id[8:] + ' ' + attack_type + ' ' + true_label \
        + ' ' + str(cm_score) + '\n')

print "Total files removed from scoring: ", n
