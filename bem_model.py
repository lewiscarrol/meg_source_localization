
import os.path as op
import os
import mne
subjects_dir = '/net/server/data/Archive/prob_learn/freesurfer'

subjects = ['301','304','307','309','312','314','313','316','322','323','324','325','326','327','328','329','331','333','334','336','340','341']



conductivity = [0.3] # for single layer        
        
for subj in subjects:

    conductivity = [0.3] # for single layer
    model = mne.make_bem_model(subject=subj, ico=5, conductivity= conductivity, subjects_dir=subjects_dir, verbose=None)

    bem = mne.make_bem_solution(model)
    
    mne.write_bem_solution('/net/server/data/Archive/prob_learn/data_processing/bem/{0}_bem.h5'.format(subj), bem, overwrite=False, verbose=None)
        
