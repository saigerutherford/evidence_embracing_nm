from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pandas as pd
from nilearn import image

atlas = datasets.fetch_atlas_smith_2009()
atlas_filename = atlas['rsn10']

correlation_measure = ConnectivityMeasure(kind='correlation')
    
sublist = [''] # strings of subject IDs to include
    
for i in range(len(sublist)):
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache', verbose=5)
    fmri_filename = '/path_to_fmri_data/derivatives/fmriprep/' + sublist[i] + '/func/' + sublist[i] + '_task-rest_run-01_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz'
    confound_file = '/path_to_fmri_data/derivatives/fmriprep/' + sublist[i] + '/func/' + sublist[i] + '_task-rest_run-01_desc-confounds_timeseries.tsv'
    confounds = pd.read_csv(confound_file, sep='\t')
    confounds.fillna(0, inplace=True)
    confounds2 = confounds[['csf','csf_derivative1','white_matter','white_matter_derivative1','trans_x','trans_x_derivative1','trans_y','trans_y_derivative1','trans_z','trans_z_derivative1',
            'rot_x','rot_x_derivative1','rot_y','rot_y_derivative1','rot_z','rot_z_derivative1']]
    time_series = masker.fit_transform(fmri_filename, confounds=confounds2)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    save_name = '/path_to_output/data_analysis/connectomes/smith10/' + sublist[i] + '.txt'
    np.savetxt(save_name, correlation_matrix)