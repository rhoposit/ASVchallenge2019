#
# Copyright 2019 CSTR, University of Edinburgh
#
# Authors: Erfan Loweimi, Joachim Fainberg, Jennifer Williams,
#          Joanna Rownicka and Ondrej Klejch
# Apache 2.0.

# creates file which maps utts to spoofing classes

import sys

utt2attack = open(sys.argv[1], 'r').readlines()
protocol = open(sys.argv[2], 'r').readlines()
utt2multi = open(sys.argv[3], 'w+')

protocol_table = [ x.strip().split(' ') for x in protocol ]
hash_table = { x.split(' ')[0].strip(): x.split(' ')[1].strip() for x in utt2attack }

for key in hash_table.keys():
    for line in protocol_table:
        if key[0:3]+key[8:] == line[1]:
            env_type = line[2]
            if len(line[3]) == 1:
                attack_type = '00'
            else:
                attack_type = line[3]
            utt2multi.write(key + ' ' + attack_type + '\n')
            break
