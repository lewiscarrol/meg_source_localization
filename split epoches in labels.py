import mne
import os
import os.path as op
import numpy as np
import pandas as pd
from mne import set_log_level

os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'

freq_range = 'beta_16_30'
rounds = [1, 2, 3, 4, 5, 6]
trial_type = ['norisk', 'prerisk', 'risk', 'postrisk']
feedback = ['positive', 'negative']
# interval of interest (1800 ms +/- 100 ms)
tmin = -0.7
tmax = 2.501
step = 0.1


#parc that we used https://balsa.wustl.edu/WN56
list_of_labels =  mne.read_labels_from_annot("fsaverage", "HCPMMP1", hemi = "both")
#print(list_of_labels)
subjects = pd.read_csv('/home/pultsinak/Рабочий стол/subj_list.csv')['subj_list'].tolist()
subjects.remove('P062')
subjects.remove('P052')
subjects.remove("P032")
subjects.remove('P045')
print(subjects)

for label in list_of_labels:
    for subj in subjects:
        for r in rounds:
            for cond in trial_type:
                for fb in feedback:
                    try:
                        epochs_num = os.listdir(
                            '/net/server/data/Archive/prob_learn/vtretyakova/sources/beta_16_30/beta_16_30_stc_fsaverage_epo/{1}_run{2}_{3}_fb_cur_{4}_{0}_fsaverage'.format(
                                freq_range, subj, r, cond, fb))
                        # print(epochs_num)
                        epo_n = int(len(epochs_num) / 2)
                        print(epo_n)
                        for ep in range(epo_n):
                            os.makedirs(
                                "/net/server/data/Archive/prob_learn/pultsinak/label_stc/split_epoches_with_labels/{0}".format(
                                    label), exist_ok=True)
                            stc = mne.read_source_estimate(
                                '/net/server/data/Archive/prob_learn/pultsinak/label_stc/{0}_stc_fsaverage_epo_label_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}/{5}_{6}'.format(
                                    freq_range, subj, r, cond, fb, ep, label))
                            # stc2 = stc.copy()
                            # print(stc)
                            sn = stc.shape[0]
                            n = stc.shape[1]
                            epochs_all_array = np.zeros(shape=(epo_n, sn, n))

                        epochs_all_array[ep, :, :] = stc.data
                        print(epochs_all_array.shape)
                        stc_epo_ave = mne.SourceEstimate(data=epochs_all_array.mean(axis=0), vertices=stc.vertices,
                                                         tmin=stc.tmin, tstep=stc.tstep)
                        print("SE Ok!")
                        stc_epo_ave.subject = subj
                        stc_epo_ave.save(
                            '/net/server/data/Archive/prob_learn/pultsinak/label_stc/split_epoches_with_labels/{0}/{1}_run{2}_{3}_fb_cur_{4}'.format(
                                label, subj, r, cond, fb))
                        print("Save OK")
                    except (OSError):
                        print('This file not exist')