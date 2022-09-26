#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:43:50 2021

import mne
import numpy as np
import pandas as pd
import os
import os.path as op



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
fif_path = '/net/server/data/Archive/prob_learn/vtretyakova/ICA_cleaned'
event_path = '/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected'


################### Merge raw files ############    
for subj in subjects:

    raws = list()
    #events = list()
    for r in rounds:
        try:
                    
            #event= np.loadtxt('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, t, fb), dtype='int')
            fif = op.join(fif_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))
            raws.append(mne.io.read_raw_fif(fif, preload=True))
            #events.append(event)
        except (OSError):
            print('This file not exist')
        first_samps = list()
        last_samps = list()
        for index in range(len(raws)):
            print(index)
        
            raws[index].pick_types(meg=True, eog=True, stim=True, eeg=False)
            first_samps.append(raws[index].first_samp)
            last_samps.append(raws[index].last_samp)
        
        raw = mne.concatenate_raws(raws, preload=True,on_mismatch='ignore')
        raw.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/raw_beta_16_30/{0}_raw.fif'.format(subj),overwrite=True)
        #event = mne.concatenate_events(events,first_samps,last_samps)
        #np.savetxt('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/events_beta_16_30/{0}_{1}_fb_cur_{2}.txt'.format(subj,t,fb),event)
        
    
s = mne.io.read_raw('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/raw_beta_16_30/P001_raw.fif')   
i = mne.find_events(s,stim_channel='STI101', shortest_event=1)  

#raw_path = '/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/raw_beta_16_30'
def read_events(filename):
    with open(filename, "r") as f:
        b = f.read().replace("[","").replace("]", "").replace("'", "")
        #b = b.split()
        #b = list(map(str.split, b))
        #b = list(map(lambda x: list(map(int, x)), b))
        return np.array(b[:]) 
########### Merge events by cond ###############

for subj in subjects:
    for t in trial_type:
        for fb in feedback:
            events = list()
            for r in rounds:
                try:
                    event= read_events('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, t, fb))
                    print(event)
                    events.append(event)
                    
                    #n = np.size(events)
                       
                    
                except (OSError):
                    print('This file not exist')
           
                 
          
            events_all = np.array(events)
            if events_all.shape == (3,):
                events_all = events_all.reshape(1,3)
            np.savetxt('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/events_beta_16_30/{0}_{1}_fb_cur_{2}.txt'.format(subj,t,fb),events_all,fmt="%s")
            
event= mne.read_epochs('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/epochs_resp/P001_run3_norisk_fb_cur_positive_raw.fif')
 
##### Merge events for baseline###########  
 
for subj in subjects:
    for t in  trial_type:
        for fb in feedback:
        
            events = list()
            for r in rounds:
                try:
                    event= mne.read_epochs('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/epochs_bl/{0}_run{3}_{1}_fb_cur_{2}.fif'.format(subj, t, fb, r))
 
                
                    events.append(event)
                    print(events)
                except (OSError):
                    print('This file not exist')
            epoches= mne.concatenate_epochs(events, on_mismatch='ignore')
            epoches.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/epochs_bl_conc/{0}_{1}_fb_cur_{2}.fif'.format(subj,t,fb))
        
        events_all = np.array(events)
        if events_all.shape == (3,):
            events_all = events_all.reshape(1,3)

        np.savetxt('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/fix_cross_mio_corr/{0}_norisk_fb_cur_{1}.txt'.format(subj,fb),events_all,fmt="%s")
    
            
temp = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run_united_fb/P001_norisk','fsaverage')

n = temp.data.shape[1] # количество временных точек (берем у донора, если донор из тех же данных.
sn = temp.data.shape[0] # sources number - количество источников (берем у донора, если донор из тех же данных).

for subj in subjects:
    for t in trial_type:
        for fb in feedback:
            data_fb = np.empty((0, sn, n))
            
                
                
            try:
                stc = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run_united_fb/{0}_{1}_fb_cur_{2}'.format(subj, t, fb), 'fsaverage').data                        
                
                stc = stc.reshape(1, sn, n) # добавляем ось блока (run)
                    
            except (OSError):
                stc = np.empty((0, sn, n))
                print('This file not exist')
                    
            data_fb = np.vstack([data_fb, stc])  # собираем все блоки в один массив 
                
            if data_fb.size != 0:
                temp.data = data_fb.mean(axis = 0)    # усредняем между блоками (run)
                temp.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run_united_fb/{0}_{1}'.format(subj, t))
            
            else:
                print('Subject has no feedbacks in this condition')
                pass    
    
for subj in subjects:
    data_fb = np.empty((0, sn, n))
    for t in trial_type:
        
        

        try:
                    ########### positive feedback #############
                    
            stc = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run/{0}_{1}_fb_cur_positive'.format(subj, t), 'fsaverage').data                        
            stc = stc.reshape(1, sn, n) # добавляем ось fb (feedback)
                    
        except (OSError):
            stc = np.empty((0, sn, n))
            print('This file not exist')
                    
        data_fb = np.vstack([data_fb, stc])  # собираем все блоки в один массив
            
                     ########### negative feedback #############
        try:
            stc = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run/{0}_{1}_fb_cur_negative'.format(subj, t), 'fsaverage').data                        
            stc = stc.reshape(1, sn, n) # добавляем ось fb (feedback)
                    
        except (OSError):
            stc = np.empty((0, sn, n))
            print('This file not exist')
             
        data_fb = np.vstack([data_fb, stc])  # собираем все блоки в один массив
        print(data_fb.shape)
            
                
    if data_fb.size != 0:
        temp.data = data_fb.mean(axis = 0)    # усредняем между positive and negative feedbacks
        temp.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/pooled_choice/{0}'.format(subj))
    else:
        print('Subject has no feedbacks in this condition')
        pass

    
for t in trial_type:
    data = np.empty((0, sn, n))
    for subj in subjects:
        try:
            stc = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/stc_merge_run_united_fb/{0}_{1}'.format(subj,t), 'fsaverage').data                        
            #stc = stc.reshape(1, sn, n) # добавляем ось блока (run)
            stc = stc.reshape(1, sn, n) # добавляем ось subj
            data = np.vstack([data, stc])  # собираем всех испытуемых в один массив        
        except (OSError):
            #stc = np.empty((0, sn, n))
            print('file for %s %s not exist'%(subj, t))
                    
        
             
          
    print('data for all subjects has shape {0}'.format(data.shape))
            
                
    if data.size != 0:
        temp.data = data.mean(axis = 0)
        temp.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/pooled_choice/{0}'.format(t))
        # усредняем между subjects
        #temp.save('/home/vtretyakova/Рабочий стол/probability_sources_new_beginning/pool_cond_empty_room_concatenate_runs/{0}'.format(t))
    else:
        print('Subject has no data in this condition')
        pass
        
temp =mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/pooled_choice/norisk-lh.stc')
n = temp.data.shape[1] # количество временных точек (берем у донора, если донор из тех же данных.
sn = temp.data.shape[0]
data = np.empty((0, sn, n))
for t in trial_type:
    
    #for subj in subjects:
        
        

    try:
                    
                    
        stc = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/pooled_choice/{0}'.format(t), 'fsaverage').data
        print(f'stc for {t} was successful download')                        
        stc = stc.reshape(1, sn, n) # добавляем ось subj
        data = np.vstack([data, stc])  # собираем все choice types в один массив        
    except (OSError):
        #stc = np.empty((0, sn, n))
        print('file for %s %s not exist'%(subj, t))
                    
        
             
          
print('data for all subjects has shape {0}'.format(data.shape))
            
                
if data.size != 0:
    temp.data = data.mean(axis = 0)    # усредняем между subjects
    temp.save('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/pooled_choice/all_ChT')
else:
    print('Subject has no data in this condition')
    pass
                
                      
                         
