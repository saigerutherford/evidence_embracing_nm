import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.stats import ttest_ind
from statsmodels.stats import multitest

# Data IO and generation
X1 = np.load('sz_z.npy')
X2 = np.load('sz_ct.npy')
y = np.load('sz_labels.npy')
n_samples, n_features = X1.shape
random_state = np.random.RandomState(0)
df5 = pd.read_csv('df5_sz_classification_groupdiff.csv')

# Group difference testing

random_state_perm = np.random.RandomState(2)
df5_perms = df5.copy(deep=False)
group = df5_perms['group']
group = np.array(group)
group_perms = shuffle(group, random_state=random_state_perm)
df5_perms['group_shuff'] = group_perms

n_permutations = 10000

z_ct_sig_diff_count_array = []
z_ct_sig_diff_count_perms_array = []

z_ct_sig_diff_count = []
z_ct_sig_diff_count_perms = []

z_sig_diff_count = []
ct_sig_diff_count = []

z_sig_diff_count_perms = []
ct_sig_diff_count_perms = []

for perm in range(0, n_permutations):
    
    random_state_perm = np.random.RandomState(perm)
    df5_perms = df5.copy(deep=False)
    group = df5_perms['group']
    group = np.array(group)
    group_perms = shuffle(group, random_state=random_state_perm)
    df5_perms['group_shuff'] = group_perms
    
    SZ = df5.query('group == 0')
    HC = df5.query('group == 1')
    
    SZ_perms = df5_perms.query('group_shuff == 0')
    HC_perms = df5_perms.query('group_shuff == 1')
    
    SZ_deviations = SZ.loc[:, SZ.columns.str.contains('Z_predict')]
    HC_deviations = HC.loc[:, HC.columns.str.contains('Z_predict')]
    
    SZ_deviations_perms = SZ_perms.loc[:, SZ_perms.columns.str.contains('Z_predict')]
    HC_deviations_perms = HC_perms.loc[:, HC_perms.columns.str.contains('Z_predict')]
    
    z_cols = SZ_deviations.columns
    
    sz_hc_pvals_z = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    sz_hc_pvals_z_perms = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    
    for index, column in enumerate(z_cols):
        test = ttest_ind(SZ_deviations[column], HC_deviations[column])
        sz_hc_pvals_z.loc[index, 'pval'] = test.pvalue
        sz_hc_pvals_z.loc[index, 'tstat'] = test.statistic
        sz_hc_pvals_z.loc[index, 'roi'] = column
        
        test_perms = ttest_ind(SZ_deviations_perms[column], HC_deviations_perms[column])
        sz_hc_pvals_z_perms.loc[index, 'pval'] = test_perms.pvalue
        sz_hc_pvals_z_perms.loc[index, 'tstat'] = test_perms.statistic
        sz_hc_pvals_z_perms.loc[index, 'roi'] = column
        
    sz_hc_fdr_z = multitest.fdrcorrection(sz_hc_pvals_z['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_z['fdr_pval'] = sz_hc_fdr_z[1]
    sz_hc_z_sig_diff = sz_hc_pvals_z.query('fdr_pval < 0.05')
    sz_hc_z_sig_diff_count = len(sz_hc_z_sig_diff)
    
    sz_hc_fdr_z_perms = multitest.fdrcorrection(sz_hc_pvals_z_perms['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_z_perms['fdr_pval'] = sz_hc_fdr_z_perms[1]
    sz_hc_z_sig_diff_perms = sz_hc_pvals_z_perms.query('fdr_pval < 0.05')
    sz_hc_z_sig_diff_count_perms = len(sz_hc_z_sig_diff_perms)
    
    SZ_cortical_thickness = SZ.loc[:, SZ.columns.str.endswith('_thickness')]
    HC_cortical_thickness = HC.loc[:, HC.columns.str.endswith('_thickness')]
    
    SZ_cortical_thickness_perms = SZ_perms.loc[:, SZ_perms.columns.str.endswith('_thickness')]
    HC_cortical_thickness_perms = HC_perms.loc[:, HC_perms.columns.str.endswith('_thickness')]
    
    ct_cols = SZ_cortical_thickness.columns
    
    sz_hc_pvals_ct = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})
    sz_hc_pvals_ct_perms = pd.DataFrame(columns={'roi','pval', 'tstat','fdr_pval'})

    for index, column in enumerate(ct_cols):
        test = ttest_ind(SZ_cortical_thickness[column], HC_cortical_thickness[column])
        sz_hc_pvals_ct.loc[index, 'pval'] = test.pvalue
        sz_hc_pvals_ct.loc[index, 'tstat'] = test.statistic
        sz_hc_pvals_ct.loc[index, 'roi'] = column
        
        test_perms = ttest_ind(SZ_cortical_thickness_perms[column], HC_cortical_thickness_perms[column])
        sz_hc_pvals_ct_perms.loc[index, 'pval'] = test_perms.pvalue
        sz_hc_pvals_ct_perms.loc[index, 'tstat'] = test_perms.statistic
        sz_hc_pvals_ct_perms.loc[index, 'roi'] = column
        
    sz_hc_fdr_ct = multitest.fdrcorrection(sz_hc_pvals_ct['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_ct['fdr_pval'] = sz_hc_fdr_ct[1]
    sz_hc_ct_sig_diff = sz_hc_pvals_ct.query('fdr_pval < 0.05')
    sz_hc_ct_sig_diff_count = len(sz_hc_ct_sig_diff)
    
    sz_hc_fdr_ct_perms = multitest.fdrcorrection(sz_hc_pvals_ct_perms['pval'], alpha=0.05, method='indep', is_sorted=False)
    sz_hc_pvals_ct_perms['fdr_pval'] = sz_hc_fdr_ct_perms[1]
    sz_hc_ct_sig_diff_perms = sz_hc_pvals_ct_perms.query('fdr_pval < 0.05')
    sz_hc_ct_sig_diff_count_perms = len(sz_hc_ct_sig_diff_perms)
    
    z_ct_sig_diff_count = sz_hc_z_sig_diff_count - sz_hc_ct_sig_diff_count
    z_ct_sig_diff_count_perms = sz_hc_z_sig_diff_count_perms - sz_hc_ct_sig_diff_count_perms
    z_ct_sig_diff_count_array.append(z_ct_sig_diff_count)
    z_ct_sig_diff_count_perms_array.append(z_ct_sig_diff_count_perms)
    
    z_sig_diff_count.append(sz_hc_z_sig_diff_count)
    ct_sig_diff_count.append(sz_hc_ct_sig_diff_count)
    z_sig_diff_count_perms.append(sz_hc_z_sig_diff_count_perms)
    ct_sig_diff_count_perms.append(sz_hc_ct_sig_diff_count_perms)

z_sig_diff_count_array = np.array(z_sig_diff_count)
ct_sig_diff_count_array = np.array(ct_sig_diff_count)
z_sig_diff_count_perms_array = np.array(z_sig_diff_count_perms)
ct_sig_diff_count_perms_array = np.array(ct_sig_diff_count_perms)
np.save('sz_cc_z_count.npy', z_sig_diff_count_array)
np.save('sz_cc_ct_count.npy', ct_sig_diff_count_array)
np.save('sz_cc_z_count_shuff.npy', z_sig_diff_count_perms_array)
np.save('sz_cc_ct_count_shuff.npy', ct_sig_diff_count_perms_array)

diff_true_case_control_test = np.array(z_ct_sig_diff_count_array)
diff_shuff_case_control_test = np.array(z_ct_sig_diff_count_perms_array)
np.save('sz_diff_true_case_control_test.npy', diff_true_case_control_test)
np.save('sz_diff_shuff_case_control_test.npy', diff_shuff_case_control_test)
pvalue_case_control_testing = (np.sum(diff_shuff_case_control_test >= diff_true_case_control_test) + 1.0) / (n_permutations + 1)
np.save('sz_pvalue_case_control_testing.npy', pvalue_case_control_testing)
