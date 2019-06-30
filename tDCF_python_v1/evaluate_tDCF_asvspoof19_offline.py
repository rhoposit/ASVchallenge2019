import os
import sys
import numpy as np
import eval_metrics as em

# Replace CM scores with your own scores or provide score file as the first argument.
cm_score_file = 'scores/cm_dev.txt'
# Replace ASV scores with organizers' scores or provide score file as the second argument.
asv_score_file = 'scores/asv_dev.txt'

args = sys.argv
if len(args) > 1:
    if len(args) != 3:
        print('USAGE: python evaluate_tDCF_asvspoof19_offline.py <CM_SCOREFILE> <ASV_SCOREFILE>')
        exit()
    else:
        cm_score_file = args[1]
        asv_score_file = args[2]

# Fix tandem detection cost function (t-DCF) parameters
Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
    'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
    'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
    'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
}

# Load organizers' ASV scores
asv_data = np.genfromtxt(asv_score_file, dtype=str)
asv_sources = asv_data[:, 0]
asv_keys = asv_data[:, 1]
asv_scores = asv_data[:, 2].astype(np.float)

# Load CM scores
cm_data = np.genfromtxt(cm_score_file, dtype=str)
cm_utt_id = cm_data[:, 0]
cm_sources = cm_data[:, 1]
cm_keys = cm_data[:, 2]
cm_scores = cm_data[:, 3].astype(np.float)

# Extract target, nontarget, and spoof scores from the ASV scores
tar_asv = asv_scores[asv_keys == 'target']
non_asv = asv_scores[asv_keys == 'nontarget']
spoof_asv = asv_scores[asv_keys == 'spoof']

# Extract bona fide (real human) and spoof scores from the CM scores
bona_cm = cm_scores[cm_keys == 'bonafide']
spoof_cm = cm_scores[cm_keys == 'spoof']

# EERs of the standalone systems and fix ASV operating point to EER threshold
eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

# Compute t-DCF
tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, False)

# Minimum t-DCF
min_tDCF_index = np.argmin(tDCF_curve)
min_tDCF = tDCF_curve[min_tDCF_index]

# Just print output
print('{}, {:.5f}, {:.5f}, {:.5f}'.format(os.path.basename(cm_score_file), eer_asv * 100, eer_cm * 100, min_tDCF))
