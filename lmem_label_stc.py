#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###At this script we create stc with the results of the LMEM for each label. Labels from HCP (Glasser) parcelations 

## Name of labels can find : https://static-content.springer.com/esm/art%3A10.1038%2Fnature18933/MediaObjects/41586_2016_BFnature18933_MOESM330_ESM.pdf

"""
Created on Fri Dec 24 10:30:22 2021

@author: kristina
"""
import mne
import mne
import os.path as op
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import copy
import statsmodels.stats.multitest as mul


os.environ['SUBJECTS_DIR'] = '/net/server/data/Archive/prob_learn/freesurfer'
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'
label_list= mne.read_labels_from_annot("fsaverage", parc = "HCPMMP1")
rh_list = mne.read_labels_from_annot("fsaverage", parc = "HCPMMP1", hemi ="rh") 
lh_list = mne.read_labels_from_annot("fsaverage", parc = "HCPMMP1", hemi = "lh")



subjects = pd.read_csv('/home/pultsinak/Рабочий стол/subj_list.csv')['subj_list'].tolist()
subjects.remove('P062') 
subjects.remove('P052') 
subjects.remove("P032")
subjects.remove('P045') 

df = pd.read_csv('/home/pultsinak/Загрузки/df_lmem_label.csv', sep = ";")

#df = pd.read_csv('/Users/kristina/Documents/stc/lmem_label/lmem_sources/exp.csv')
df = df.drop(df.columns[0], axis = 1)
labels= pd.DataFrame(df.label_short.tolist())

#labels = labels['label_short'].tolist()


#intervals = [[-0.700,-0.600],[-0.6,-0.5], [-0.5, -0.4], [-0.4, -0.3], [-0.3, -0.2], 
             #[-0.2, -0.1], [-0.1, 0.0], [0.0, 0.1], [0.1, 0.2], [0.2, 0.3],
             #[0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.66, 0.7], [0.7, 0.8],
             #[0.8, 0.9], [0.900, 1.000], [1.0, 1.1], [1.1, 1.2],[1.2, 1.3],
            # [1.3, 1.4], [1.4, 1.5], [1.5, 1.6], [1.6, 1.7], [1.7, 1.8], [1.8, 1.9],
            # [1.9, 2.0], [2.0, 2.1], [2.1, 2.2], [2.2, 2.3], [2.3, 2.4]]


tmin = [ -0.700,-0.600, -0.500,-0.400,-0.300,-0.200, -0.100, 0.000, 0.100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,1.3, 1.4, 1.5,1.6, 1.7, 1.8,1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
tmax = [ -0.600,-0.500, -0.400,-0.300,-0.200,-0.100,0.000, 0.100, 0.200, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7, 1.8, 1.9,2.0, 2.1, 2.2, 2.3,2.4, 2.5]
tstep = 0.1




#label_names = [label.name for label in rh_list]
#label_hemis = [label.hemi for label in label_list]






stc_test = mne.read_source_estimate('/net/server/data/Archive/prob_learn/pultsinak/sources_sLoreta/beta_16_30/morphed_stc_epo_var2/P001_run2_norisk_fb_cur_negative_fsaverage/0', 'fsaverage').crop(tmin= -0.700, tmax= 2.500)
stc_test.resample(10)


for l, r in zip(lh_list, rh_list):
    
    
    labell = mne.BiHemiLabel(l,r, name = l.name)
    
    stc_l = stc_test.in_label(labell)
    vertices = stc_l.vertices 
    
    v_lh = np.array (vertices[0], dtype = "int32")
    v_rh = np.array (vertices[1], dtype ="int32")
    vert = []
    vert.append(v_lh)    
    vert.append(v_rh) 
    
    
    stc_lh_data = stc_l.lh_data
    r_lh = stc_lh_data.shape[0]
    #pval_in_intevals_lh = np.empty((r_lh, 32))
    pval_s_lh = df.loc[df['label_short'] == l.name] 
    pval_trial_type_lh = pval_s_lh['trial_type'].tolist()
    pval_in_intevals_lh = []
    pval_in_intevals_lh.append(pval_trial_type_lh)
    p_lh = np.array(pval_in_intevals_lh)
    
    lh_data = np.repeat(p_lh, r_lh, axis = 0)
    print(l.name)
    
    
    stc_rh_data = stc_l.rh_data
    r_rh = stc_rh_data.shape[0]
    #pval_in_intevals_lh = np.empty((r_rh, 32))
    
    pval_s_rh = df.loc[df['label_short'] == r.name] 
    pval_trial_type_rh = pval_s_rh['trial_type'].tolist()
    #
    pval_in_intevals_rh = [] 
    pval_in_intevals_rh.append(pval_trial_type_rh)
    p_rh = np.array(pval_in_intevals_rh)
    print(r.name)
    rh_data = np.repeat(p_rh, r_rh, axis = 0)
    
        
        
    data = np.vstack([lh_data, rh_data])
        
        #data = np.array(lh_data, rh_data)
        #data.append(pval_for_stc)
        
        
    print("pval_for_stc")
        
    stc_label = mne.SourceEstimate(data = data, vertices = vertices, tmin = stc_test.tmin, tstep = stc_test.tstep)
    print("stc_label")
    stc_label.subject = "fsaverage"
    stc_label.save('/net/server/data/Archive/prob_learn/pultsinak/label_stc/label_stc_with_LMEM/trial_type_{0}'.format(r.name))
    
    
    
   

#stc = mne.read_source_estimate("/net/server/data/Archive/prob_learn/pultsinak/label_stc/label_stc_with_LMEM/trial_type_R_1_ROI-rh.stc")

#stc_label.plot()


#a = data[:len(vertices[0])]
#b = data[-len(vertices[1]):]











        
        
        


