import os
import mne
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import multitest as mul
import matplotlib.pyplot as plt

os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'


label_dir = "/net/server/data/Archive/prob_learn/pultsinak/label_stc/functional_labels(Yao)"
labels = [
  #  ['motor_left-rh.label'],
#    ['motor_right-lh.label'],
 #['precuneus-lh.label', 'precuneus-rh.label'],
 #   ['wernicke-lh.label'],
 #   ['parahippocampal_gyrus_right-rh.label'],
   #['Posterior cingulum-lh.label', 'Posterior cingulum-rh.label'],
  #  ['parahippocampal_gyrus-lh.label', 'parahippocampal_gyrus-rh.label'],
 #   ['heschl_gyrus-lh.label'],
 #   ['anterior_cingulate-lh.label', 'anterior_cingulate-rh.label'],
  #['SMA-lh.label', 'SMA-rh.label'],
   # ['parahippocampal_gyrus_left-lh.label'],
  #  ['middle_cingulate-lh.label', 'middle_cingulate-rh.label'],
 #   ['broca-lh.label'],
   # ['SCEF-lh.label', 'SCEF-lh.label'],
#['new_posterior_cingulate-lh.label', 'new_posterior_cingulate-rh.label'],
 ##  ['6ma-lh.label', '6ma-rh.label'],
 #['6mp-lh.label', '6mp-rh.label'],
 ['6ma-lh.label', '6ma-rh.label'],
  #  ['6mp-lh.label', '6mp-rh.label'],
  #  ['23c-lh.label', '23c-rh.label'],
  #  ['24dd-lh.label', '24dd-rh.label'],
 #   ['24dv-lh.label', '24dv-rh.label'],
  #  ['a32pr-lh.label', 'a32pr-rh.label'],
 #   ['new_posterior_cingulate-lh.label', ['new_posterior_cingulate-rh.label'],
#    ['p32pr-lh.label', 'p32pr-rh.label'],
 #   ['SCEF-lh.label', 'SCEF-rh.label'],
#["L_9-46d_ROI-lh.label"]
#['8BL-lh.label'], ['8BL-rh.label'],
#['8BM-lh.label'], ['8BM-rh.label']
 #['sts-lh.label'],
#['frontaloperculum-lh.label'],
#['ventrallateralprefrontalcortex-lh.label'],
#['dorsolateralprefrontalcortex-lh.label', "dorsolateralprefrontalcortex-rh.label"],
#['retrosplenial-lh.label'],
#['presma_acc-lh.label']
#["superiorfrontal-lh.label"]
#["sf_dorsal-lh.label"],
#["sf_med1-lh.label"],
#["sf_med2-lh.label"],
#["sf_med3-lh.label"]
]



data_dir= "/net/server/data/Archive/prob_learn/vtretyakova/sources/beta_16_30/beta_16_30_stc_fsaverage_ave_into_subj"
os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/label_stc/ttest_pos_vs_neg', exist_ok = True)
freq_range = 'beta_16_30'
rounds = [1, 2, 3, 4, 5, 6]
session = ["risk_fb_cur_positive", "risk_fb_cur_negative"]
trial_type = ['norisk', 'prerisk', 'risk', 'postrisk']
feedback = ['positive', 'negative']
# interval of interest (1800 ms +/- 100 ms)
tmin = -0.7
tmax = 2.501
step = 0.1
inter = [-0.800, 2.400]

def signed_p_val(t, p_val):
    if t >= 0:
        return 1 - p_val
    else:
        return -(1 - p_val)

stc_test = mne.read_source_estimate('/net/server/data/Archive/prob_learn/vtretyakova/sources/beta_16_30/beta_16_30_stc_fsaverage_ave_into_subj/P001_risk_fb_cur_negative', 'fsaverage').crop(tmin=inter[0], tmax=inter[1], include_tmax=True)

for t in trial_type:
    print(t)
    for label in labels:
        data_dir = "/net/server/data/Archive/prob_learn/vtretyakova/sources/beta_16_30/beta_16_30_stc_fsaverage_ave_into_subj"
        stc_test = mne.read_source_estimate('/net/server/data/Archive/prob_learn/vtretyakova/sources/beta_16_30/beta_16_30_stc_fsaverage_ave_into_subj/P001_risk_fb_cur_negative',
            'fsaverage').crop(tmin=inter[0], tmax=inter[1], include_tmax=True)
        # stc_test.resample(20)
        time = stc_test.times

        os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/label_stc/ttest_pos_vs_neg', exist_ok=True)
        if len(label) == 2:
            print(label[0], label[1])
            lh = mne.read_label(os.path.join(label_dir, label[0]))
            rh = mne.read_label(os.path.join(label_dir, label[1]))
            label_both = mne.BiHemiLabel(lh, rh)
            stc_label = stc_test.in_label(label_both)
            comp1_per_sub = np.zeros(shape=(len(subjects), stc_label.data.shape[0], stc_label.data.shape[1]))
            # print(comp1_per_sub.shape)
            comp2_per_sub = np.zeros(shape=(len(subjects), stc_label.data.shape[0], stc_label.data.shape[1]))
            # print(comp2_per_sub.shape)

        for ind, subj in enumerate(subjects):
            print(ind + 1, subj)
            temp1 = mne.read_source_estimate(
                os.path.join(data_dir, "{0}_{1}_fb_cur_{2}-lh.stc".format(subj, t, feedback[0]))).crop(tmin=inter[0],
                                                                                                       tmax=inter[1],
                                                                                                       include_tmax=True)
            # temp1.resample(10)
            temp11 = temp1.copy().in_label(label_both)#.data.mean(axis=0)
            comp1_per_sub[ind, :, :] = temp11

            temp2 = mne.read_source_estimate(os.path.join(data_dir, "{0}_{1}_fb_cur_{2}-lh.stc".format(subj, t, feedback[1]))).crop(tmin=inter[0], tmax=inter[1],include_tmax=True)

            # temp2.resample(10)
            temp22 = temp2.copy().in_label(label_both)#.data.mean(axis=0) #add mean for averiging in hole label
            comp2_per_sub[ind, :, :] = temp22

        print("calculation ttest")
        t_stat, p_val = stats.ttest_rel(comp2_per_sub, comp1_per_sub, axis=0)

        # width, height = p_val.shape
        # p_val_resh = p_val.reshape(width * height)
        # _, p_val = mul.fdrcorrection(p_val_resh)
        # p_val = p_val.reshape((width, height))

        # t_stat_1, p_val_1 = stats.ttest_1samp(comp1_per_sub, popmean=0, axis=0)
        # t_stat_2, p_val_2 = stats.ttest_1samp(comp2_per_sub, popmean=0, axis=0)
        print(p_val.min(), p_val.mean(), p_val.max())
        print(t_stat.min(), t_stat.mean(), t_stat.max())

        stc_label.data = p_val
        print(stc_test)

        stc_label.save('/net/server/data/Archive/prob_learn/pultsinak/label_stc/ttest_pos_vs_neg/{0}_fb_cur_pos_vs_neg_pval_full_fdr'.format(t))
l_stc = mne.read_source_estimate("/net/server/data/Archive/prob_learn/pultsinak/label_stc/ttest_pos_vs_neg/risk_fb_cur_pos_vs_neg_pval_full_fdr-rh.stc")
l_stc.subject = "fsaverage"
l_stc.plot(hemi="both")
