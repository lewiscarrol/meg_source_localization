import mne
import os
import pandas as pd
import numpy as np

os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'

subjects = pd.read_csv('/home/pultsinak/Рабочий стол/subj_list.csv')['subj_list'].tolist()
subjects.remove('P062')
subjects.remove('P052')
subjects.remove("P032")
subjects.remove('P045')

rounds = [1, 2, 3, 4, 5, 6]
freq_range = "beta_16_30"
trial_type = ['norisk', 'prerisk', 'risk', 'postrisk']
feedback = ['positive', 'negative']
tmin = -0.700
tmax = 2.501
step = 0.1


#parc that we used https://balsa.wustl.edu/WN56
labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", hemi = "both")

os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/label_stc/{0}_stc_fsaverage_epo_label_var2'.format(freq_range), exist_ok = True)

for subj in subjects:
    for r in rounds:
        for cond in trial_type:
            for fb in feedback:
                try:
                    epochs_num = os.listdir(
                        '/net/server/data/Archive/prob_learn/vtretyakova/sources/{0}/beta_16_30_stc_epo_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}'.format(
                            freq_range, subj, r, cond, fb))
                    print(int(len(epochs_num) / 2))
                    os.makedirs(
                        '/net/server/data/Archive/prob_learn/pultsinak/label_stc/{0}_stc_fsaverage_epo_label_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}'.format(
                            freq_range, subj, r, cond, fb))

                    for ep in range(int(len(epochs_num) / 2)):
                        stc = mne.read_source_estimate(
                            "/net/server/data/Archive/prob_learn/vtretyakova/sources/{0}/beta_16_30_stc_epo_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}/{5}".format(
                                freq_range, subj, r, cond, fb, ep))
                        stc_2 = stc.copy()
                        for label in labels:
                            stc_label = stc_2.in_label(label)
                            stc_label.save(
                                '/net/server/data/Archive/prob_learn/pultsinak/label_stc/{0}_stc_fsaverage_epo_label_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}/{5}_{6}'.format(
                                    freq_range, subj, r, cond, fb, ep, label))
                except (OSError):
                    print('This file not exist')
