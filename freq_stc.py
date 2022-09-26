#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import mne
import os
import os.path as op
import numpy as np
import pandas as pd
from functions import make_freq_stc, make_stc_epochs_from_freq_epochs, make_stc_epochs_from_freq_epochs_var2


L_freq = 16
H_freq = 31
f_step = 2

#time_bandwidth = 4 #(by default = 4)
# if delta (1 - 4 Hz) 
#n_cycles = np.array([1, 1, 1, 2]) # уточнить


period_start = -1.750
period_end = 2.750

baseline = (-0.35, -0.05)

freq_range = 'beta_16_30'

description = 'STC for beta power 16 - 30 Hz, epochs, make with **source_band_induced_power**, but substituting the epochs into the function one at time, **early log** method'

# This code sets an environment variable called SUBJECTS_DIR
os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'

# 40 subjects with all choice types

subjects = pd.read_csv('/home/vtretyakova/Рабочий стол/probability_learning/sources/subj_list.csv')['subj_list'].tolist()

subjects.remove('P062') #without MRI
subjects.remove('P052') # bad segmentation, попробовали запустить freesurfer еще раз не помогло

subjects.remove('P032') #ValueError: dimension mismatch - попробовали запустить freesurfer еще раз не помогло. Надо разбираться

subjects.remove('P045') #RuntimeError: Could not find neighbor for vertex 4395 / 10242 когда использую для задания пространства источников src, ico5 вместо oct6


#subjects = subjects[22:] # from P033


rounds = [1, 2, 3, 4, 5, 6]

trial_type = ['norisk', 'prerisk', 'risk', 'postrisk']


feedback = ['positive', 'negative']

data_path = '/net/server/data/Archive/prob_learn/vtretyakova/ICA_cleaned'
os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/sources/{0}'.format(freq_range), exist_ok = True)
os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/sources/{0}/{0}_stc_epo_var2'.format(freq_range), exist_ok = True)

########## function for extract freq for each epochs (return stcs)########
def make_stc_epochs_from_freq_epochs(subj, r, cond, fb, data_path, L_freq, H_freq, f_step, period_start, period_end, baseline, bem, src):

    bands = dict(beta=[L_freq, H_freq])
    #freqs = np.arange(L_freq, H_freq, f_step)
    
    #read events
	#events for baseline
	# download marks of positive feedback
	
    events_pos = np.loadtxt("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_positive_fix_cross.txt".format(subj, r), dtype='int') 
    

        # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_pos.shape == (3,):
        events_pos = events_pos.reshape(1,3)
        
    # download marks of negative feedback      
    
    events_neg = np.loadtxt("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_negative_fix_cross.txt".format(subj, r), dtype='int')
    
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_neg.shape == (3,):
        events_neg = events_neg.reshape(1,3) 
    
    #объединяем негативные и позитивные фидбеки для получения общего бейзлайна по ним, и сортируем массив, чтобы времена меток шли в порядке возрастания    
    events = np.vstack([events_pos, events_neg])
    events = np.sort(events, axis = 0) 
    
    #events, which we need
    events_response = np.loadtxt('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb), dtype='int')
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
    if events_response.shape == (3,):
        events_response = events_response.reshape(1,3)
    
	           
    raw_fname = op.join(data_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))

    raw_data = mne.io.Raw(raw_fname, preload=True)
        
    
    picks = mne.pick_types(raw_data.info, meg = True, eog = True)
		    
	# Forward Model
    trans = '/net/server/mnt/Archive/prob_learn/freesurfer/{0}/mri/T1-neuromag/sets/{0}-COR.fif'.format(subj)
        
	   	    
    #epochs for baseline
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs_bl = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, baseline = None, picks = picks, preload = True)
    cov = mne.compute_covariance(epochs=epochs_bl, method='auto', tmin=-0.35, tmax = -0.05)
     
    epochs_bl.resample(100)
    
    ####### ДЛЯ ДАННЫХ ##############
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs = mne.Epochs(raw_data, events_response, event_id = None, tmin = period_start, 
		                tmax = period_end, baseline = None, picks = picks, preload = True)


                
    fwd = mne.make_forward_solution(info=epochs.info, trans=trans, src=src, bem=bem)	                
    inv = mne.minimum_norm.make_inverse_operator(raw_data.info, fwd, cov, loose=0.2) 	                
		       
    epochs.resample(100) 
    
    stc_baseline = mne.minimum_norm.source_band_induced_power(epochs_bl.pick('meg'), inv, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"].crop(tmin=baseline[0], tmax=baseline[1], include_tmax=True)
    
    
    #усредняем по времени
    b_line  = stc_baseline.data.mean(axis=-1)
    b_line = 10*np.log10(b_line)
    
    stc_epo_list = []
    for ix in range(len(epochs)):
        stc = mne.minimum_norm.source_band_induced_power(epochs[ix].pick('meg'), inv, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"]
        data = 10*np.log10(stc.data)
        #stc.data = 10*np.log10(stc.data/b_line[:, np.newaxis])
        stc.data = data - b_line[:, np.newaxis]
        stc_epo_list.append(stc)
    return (stc_epo_list)


############## function if you need to extract stc with freq avereged by epochs ##########
def make_freq_stc(subj, r, cond, fb, data_path, L_freq, H_freq, f_step, period_start, period_end, baseline, bem, src):
    
    bands = dict(beta=[L_freq, H_freq])
    #freqs = np.arange(L_freq, H_freq, f_step)
    
    #read events
	#events for baseline
	# download marks of positive feedback
	
    events_pos = np.loadtxt("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_positive_fix_cross.txt".format(subj, r), dtype='int') 
    

        # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_pos.shape == (3,):
        events_pos = events_pos.reshape(1,3)
        
    # download marks of negative feedback      
    
    events_neg = np.loadtxt("/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_negative_fix_cross.txt".format(subj, r), dtype='int')
    
    
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
    if events_neg.shape == (3,):
        events_neg = events_neg.reshape(1,3) 
    
    #объединяем негативные и позитивные фидбеки для получения общего бейзлайна по ним, и сортируем массив, чтобы времена меток шли в порядке возрастания    
    events = np.vstack([events_pos, events_neg])
    events = np.sort(events, axis = 0) 
    
    #events, which we need
    events_response = np.loadtxt('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/events_by_cond_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb), dtype='int')
    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
    if events_response.shape == (3,):
        events_response = events_response.reshape(1,3)
    
	           
    raw_fname = op.join(data_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))

    raw_data = mne.io.Raw(raw_fname, preload=True)
        
    
    picks = mne.pick_types(raw_data.info, meg = True, eog = True)
		    
	# Forward Model
    trans = '/net/server/mnt/Archive/prob_learn/freesurfer/{0}/mri/T1-neuromag/sets/{0}-COR.fif'.format(subj)
        
	   	    
    #epochs for baseline
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs_bl = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1.0, baseline = None, picks = picks, preload = True)
    cov = mne.compute_covariance(epochs=epochs_bl, method='auto', tmin=-0.35, tmax = -0.05)
     
    epochs_bl.resample(100)
    
    ####### ДЛЯ ДАННЫХ ##############
    # baseline = None, чтобы не вычитался дефолтный бейзлайн
    epochs = mne.Epochs(raw_data, events_response, event_id = None, tmin = period_start, 
		                tmax = period_end, baseline = None, picks = picks, preload = True)


                
    fwd = mne.make_forward_solution(info=epochs.info, trans=trans, src=src, bem=bem)	                
    inv = mne.minimum_norm.make_inverse_operator(raw_data.info, fwd, cov, loose=0.2) 	                
		       
    epochs.resample(100) 
    
    stc_baseline = mne.minimum_norm.source_band_induced_power(epochs_bl.pick('meg'), inv, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"].crop(tmin=baseline[0], tmax=baseline[1], include_tmax=True)
    
    
    #усредняем по времени
    b_line  = stc_baseline.data.mean(axis=-1)
    
    stc = mne.minimum_norm.source_band_induced_power(epochs.pick('meg'), inv, bands, use_fft=False, df = f_step, n_cycles = 8)["beta"]
    
    stc.data = 10*np.log10(stc.data/b_line[:, np.newaxis])
    # check beta power on norisk
    return (stc)
    #morph = mne.compute_source_morph(stc, subject_from=subj, subject_to='fsaverage')
    #stc_fsaverage = morph.apply(stc)

    #return (stc_fsaverage)
    
    
    

##############################################################################################################


for subj in subjects:
    bem = mne.read_bem_solution('/net/server/data/Archive/prob_learn/dataprocessing/bem/{0}_bem.h5'.format(subj), verbose=None)
    src = mne.setup_source_space(subject =subj, spacing='ico5', add_dist=False ) # by default - spacing='oct6' (4098 sources per hemisphere)
    for r in rounds:
        for cond in trial_type:
            for fb in feedback:

                try:

                    
                    stc_epo_list = make_stc_epochs_from_freq_epochs(subj, r, cond, fb, data_path, L_freq, H_freq, f_step, period_start, period_end, baseline, bem, src)
                    print('Количество эпох %s' % len(stc_epo_list))
                    
                    os.makedirs('/net/server/data/Archive/prob_learn/pultsinak/sources/{0}/{0}_stc_epo_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}'.format(freq_range, subj, r, cond, fb))
                    
                    for s in range(len(stc_epo_list)):
                        stc_epo_list[s].save('/net/server/data/Archive/prob_learn/pultsinak/sources/{0}/{0}_stc_epo_var2/{1}_run{2}_{3}_fb_cur_{4}_{0}/{5}'.format(freq_range, subj, r, cond, fb, s))
                    
                                
         
                except (OSError):
                    print('This file not exist')

