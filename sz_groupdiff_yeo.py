import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.stats import ttest_ind
from statsmodels.stats import multitest


# Data IO and generation
X1 = np.load('sz_yeo_z.npy')
X4_resid = np.load('sz_yeo_resid.npy')
y = np.load('sz_yeo_labels.npy')
n_samples, n_features = X1.shape
random_state = np.random.RandomState(0)
df = pd.read_csv('df_sz_classification_groupdiff.csv')
yeo_data_only_resid = pd.read_csv('yeo_data_only_resid.csv')

# Group difference testing

n_permutations = 10000

z_ct_sig_diff_count_perms_array = []
z_ct_sig_diff_count_perms = []
z_sig_diff_count_perms = []
ct_sig_diff_count_perms = []

for perm in range(0, n_permutations):
    
    random_state_perm = np.random.RandomState(perm)
    df_perms = df.copy(deep=False)
    group = df_perms['group']
    group = np.array(group)
    group_perms = shuffle(group, random_state=random_state_perm)
    df_perms['group_shuff'] = group_perms
    
    SZ = df.query('group == 0')
    HC = df.query('group == 1')
    
    SZ_perms = df_perms.query('group_shuff == 0')
    HC_perms = df_perms.query('group_shuff == 1')

    SZ_deviations_perms = SZ_perms.loc[:, SZ_perms.columns.str.contains('Z_predict')]
    HC_deviations_perms = HC_perms.loc[:, HC_perms.columns.str.contains('Z_predict')]
    
    z_cols = SZ_deviations_perms.columns
    
    sz_hc_pvals_z_perms = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    
    for index, column in enumerate(z_cols):
        test_perms = ttest_ind(SZ_deviations_perms[column], HC_deviations_perms[column])
        sz_hc_pvals_z_perms.loc[index, 'pval'] = test_perms.pvalue
        sz_hc_pvals_z_perms.loc[index, 'tstat'] = test_perms.statistic
        sz_hc_pvals_z_perms.loc[index, 'roi'] = column
    
    sz_hc_fdr_z_perms = multitest.fdrcorrection(sz_hc_pvals_z_perms['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_z_perms['fdr_pval'] = sz_hc_fdr_z_perms[1]
    sz_hc_z_sig_diff_perms = sz_hc_pvals_z_perms.query('fdr_pval < 0.05')
    sz_hc_z_sig_diff_count_perms = len(sz_hc_z_sig_diff_perms)

    SZ_cortical_thickness_perms = SZ_perms.loc[:, SZ_perms.columns.str.endswith('_thickness')]
    HC_cortical_thickness_perms = HC_perms.loc[:, HC_perms.columns.str.endswith('_thickness')]
    
    ct_cols = SZ_cortical_thickness_perms.columns
    
    sz_hc_pvals_ct_perms = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})

    for index, column in enumerate(ct_cols):
        
        test_perms = ttest_ind(SZ_cortical_thickness_perms[column], HC_cortical_thickness_perms[column])
        sz_hc_pvals_ct_perms.loc[index, 'pval'] = test_perms.pvalue
        sz_hc_pvals_ct_perms.loc[index, 'tstat'] = test_perms.statistic
        sz_hc_pvals_ct_perms.loc[index, 'roi'] = column
    
    sz_hc_fdr_ct_perms = multitest.fdrcorrection(sz_hc_pvals_ct_perms['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_ct_perms['fdr_pval'] = sz_hc_fdr_ct_perms[1]
    sz_hc_ct_sig_diff_perms = sz_hc_pvals_ct_perms.query('fdr_pval < 0.05')
    sz_hc_ct_sig_diff_count_perms = len(sz_hc_ct_sig_diff_perms)
    
    z_ct_sig_diff_count_perms = sz_hc_z_sig_diff_count_perms - sz_hc_ct_sig_diff_count_perms
    z_ct_sig_diff_count_perms_array.append(z_ct_sig_diff_count_perms)
    
    z_sig_diff_count_perms.append(sz_hc_z_sig_diff_count_perms)
    ct_sig_diff_count_perms.append(sz_hc_ct_sig_diff_count_perms)

z_sig_diff_count_perms_array = np.array(z_sig_diff_count_perms)
ct_sig_diff_count_perms_array = np.array(ct_sig_diff_count_perms)
diff_shuff_case_control_test = np.array(z_ct_sig_diff_count_perms_array)
np.save('sz_z_count_shuff.npy', z_sig_diff_count_perms_array)
np.save('sz_ct_count_shuff.npy', ct_sig_diff_count_perms_array)
np.save('sz_diff_shuff_case_control_test.npy', diff_shuff_case_control_test)