#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:43:50 2021

@author: pultsinak
"""

# -*- coding: utf-8 -*-


import mne
import numpy as np
import pandas as pd
import os


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

scheme = pd.read_csv('/home/pultsinak/Рабочий стол/SCHEMES2.csv')
scheme= scheme.loc[222:]

intervals =[[-0.800, -0.700],[-0.700, -0.600],[-0.600,-0.500], [-0.500,-0.400], [-0.400,-0.300],
            [-0.300,-0.200],[-0.200,-0.100], [-0.100, 0.0], [0.0, 0.100],[0.100, 0.200],[0.200,0.300],
            [0.300,0.400],[0.400,0.500],[0.500, 0.600],[0.600,0.700],[0.700, 0.800],[0.800,0.900],[0.900, 1.000],
            [1.000, 1.100],[1.100,1.200],[1.200,1.300],[1.300,1.400],[1.400,1.500],[1.500,1.600],[1.600,1.700],
            [1.700,1.800],[1.800,1.900],[1.900,2.000], [2.000,2.100],[2.100,2.300],[2.300,2.400],[2.400,2.501]]
#parc that we used https://balsa.wustl.edu/WN56
labels =  mne.read_labels_from_annot("fsaverage", "HCPMMP1", hemi = "both")
#labels = list(labels)
#print(labels)
data_dir = "/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/beta_16_30/morphed_stc_epo_var2"
# crop stc by time

lines = ["freq_range = {}".format(freq_range), "rounds = {}".format(rounds), "trial_type = {}".format(trial_type), "feedback = {}".format(feedback), "tmin = {}".format(tmin), "tmax = {}".format(tmax), "step = {} усредение сигнала +/- 1,0 step от значения над topomap  ".format(step)]


with open("/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/df_for_lmem/config.txt","w") as file:
    for  line in lines:
        file.write(line + '\n')
        
def make_subjects_df(label_stc, epo_n, subj, r, t, fb, tmin, tmax, step, scheme):
    time_intervals = np.arange(tmin, tmax, step)
    list_of_time_intervals = []
    i = 0
    while i < (len(time_intervals) - 1):
        interval = time_intervals[i:i+2]
        list_of_time_intervals.append(interval)
        #print(interval)
        i = i+1
    
    list_of_beta_power = []    
    for i in list_of_time_intervals:
        label_in_interval = label_stc.copy()
        label_in_interval= label_in_interval.crop(tmin=i[0], tmax=i[1], include_tmax=True)
        #combined_planar_in_interval = np.mean(combined_planar_in_interval.data, axis=0)
        #print(combined_planar_in_interval)
        
        mean_label_stc = np.mean(label_in_interval.data, axis=-1)
        
        
        #print(label_in_interval.shape)
        mean_label = np.mean( mean_label_stc .data, axis = 0)
        
    
        beta_power = []

        for j in range(epo_n):
            a = mean_label
            beta_power.append(a)
        list_of_beta_power.append(beta_power)
    
    trial_number = range(epo_n)
    
    subject = [subj]*len(range(epo_n))
    run = ['run{0}'.format(r)]*len(range(epo_n))
    trial_type = [t]*len(range(epo_n))
    #feedback_cur = [fb]*len(range(epo_n)
    feedback_cur = [fb]*len(range(epo_n))
    
    
    feedback_prev_data = np.loadtxt("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/prev_fb_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}_prev_fb.txt".format(subj, r, t, fb), dtype='int')
    if feedback_prev_data.shape == (3,):
        feedback_prev_data = feedback_prev_data.reshape(1,3)
        
    
    
    FB_rew = [50, 51]
    FB_lose = [52, 53]

    feedback_prev = []
    for i in feedback_prev_data:
        if i[2] in FB_rew:
            a = 'positive'
            
        else:
            a = 'negative'
            
        feedback_prev.append(a)   
        
    # схема подкрепления   
    a = scheme.loc[(scheme['fname'] == subj) & (scheme['block'] == r)]['scheme'].tolist()[0]
    sch = [a]*len(range(epo_n))    
    
    
    df = pd.DataFrame()
    
    
    df['trial_number'] = trial_number
    
    # beta на интервалах
    for idx, beta in enumerate(list_of_beta_power):
        df['beta power %s'%list_of_time_intervals[idx]] = beta
    

    #df['beta_power'] = beta_power
    df['subject'] = subject
    df['round'] = run
    df['trial_type'] = trial_type
    df['feedback_cur'] = feedback_cur
    df['feedback_prev'] = feedback_prev
    df['scheme'] = sch
        
    return (df)



#os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/label_array', exist_ok = True)
for label in labels:
    print(label)
    df = pd.DataFrame()
    #os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/df_for_lmem/{0}'.format(label),exist_ok = True)
    for subj in subjects:
        for r in rounds:
            for t in trial_type:
                for fb in feedback:
                    
                    try:
                        epochs_num = os.listdir('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/beta_16_30/morphed_stc_epo_var2/{0}_run{1}_{2}_fb_cur_{3}_fsaverage'.format(subj, r, t, fb))
                        epo_n = (int(len(epochs_num) / 2))
                      
                        #os.makedirs("/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/df_for_lmem/{0}".format(label))
                        for ep in range(epo_n):
                            stc = mne.read_source_estimate("/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/beta_16_30/morphed_stc_epo_var2/{0}_run{1}_{2}_fb_cur_{3}_fsaverage/{4}".format(subj, r, t, fb, ep))
                            stc2 = stc.copy()
                            #info = stc2.info
                            label_stc = stc2.in_label(label)
                            
                            df_subj = make_subjects_df(label_stc, epo_n, subj, r, t, fb, tmin, tmax, step,scheme)
                            df = df.append(df_subj)            
                    except (OSError, FileNotFoundError):
                        print('This file not exist')
    df.to_csv('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/df_for_lmem/{0}csv'.format(label))
                            
                            
                      
                         