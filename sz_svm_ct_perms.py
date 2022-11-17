import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from scipy.stats import ttest_ind
from statsmodels.stats import multitest

# Classification

# Data IO and generation
X1 = np.load('sz_z.npy')
X4_resid = np.load('sz_ct_age_sex_site.npy')
y = np.load('sz_labels_ct.npy')
n_samples, n_features = X1.shape
random_state = np.random.RandomState(0)

df = pd.read_csv('df_sz_classification_groupdiff.csv')

cv = StratifiedKFold(n_splits=10)
n_permutations = 1000

classifier_z = svm.SVC(kernel="linear", probability=True, random_state=random_state)
classifier_z2 = svm.SVC(kernel="linear", probability=True, random_state=random_state)

tprs_z = []
aucs_z = []
tprs_z2 = []
aucs_z2 = []
aucs_z_perms = []
mean_auc_z_perms = []
aucs_z_perms_shuff = []
mean_auc_z_perms_shuff = []
mean_fpr_z = np.linspace(0, 1, 100)
mean_fpr_z2 = np.linspace(0, 1, 100)

classifier_ct = svm.SVC(kernel="linear", probability=True, random_state=random_state)
classifier_ct2 = svm.SVC(kernel="linear", probability=True, random_state=random_state)

tprs_ct = []
aucs_ct = []
tprs_ct2 = []
aucs_ct2 = []
aucs_ct_perms = []
mean_auc_ct_perms = []
aucs_ct_perms_shuff = []
mean_auc_ct_perms_shuff = []
mean_fpr_ct = np.linspace(0, 1, 100)
mean_fpr_ct2 = np.linspace(0, 1, 100)

diff_mean_auc_z_ct_perms = []
diff_mean_auc_z_ct_perms_shuff = []

fig, ax = plt.subplots(figsize=(15,15))

for perm in range(0, n_permutations):
    random_state_perm = np.random.RandomState(perm)
    y_perms = shuffle(y, random_state=random_state_perm)
    
    for i, (train, test) in enumerate(cv.split(X1, y_perms)):
        classifier_z2.fit(X1[train], y_perms[train])
        viz = RocCurveDisplay.from_estimator(
            classifier_z2,
            X1[test],
            y_perms[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr_z2 = np.interp(mean_fpr_z2, viz.fpr, viz.tpr)
        interp_tpr_z2[0] = 0.0
        tprs_z2.append(interp_tpr_z2)
        aucs_z2.append(viz.roc_auc)

    mean_tpr_z2 = np.mean(tprs_z2, axis=0)
    mean_tpr_z2[-1] = 1.0
    mean_auc_z2 = auc(mean_fpr_z2, mean_tpr_z2)
    std_auc_z2 = np.std(aucs_z2)
    aucs_z_perms_shuff.append(aucs_z2)
    mean_auc_z_perms_shuff.append(mean_auc_z2)
    
    for i, (train, test) in enumerate(cv.split(X4_resid, y_perms)):
        classifier_ct2.fit(X4_resid[train], y_perms[train])
        viz = RocCurveDisplay.from_estimator(
            classifier_ct2,
            X4_resid[test],
            y_perms[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr_ct2 = np.interp(mean_fpr_ct2, viz.fpr, viz.tpr)
        interp_tpr_ct2[0] = 0.0
        tprs_ct2.append(interp_tpr_ct2)
        aucs_ct2.append(viz.roc_auc)

    mean_tpr_ct2 = np.mean(tprs_ct2, axis=0)
    mean_tpr_ct2[-1] = 1.0
    mean_auc_ct2 = auc(mean_fpr_ct2, mean_tpr_ct2)
    std_auc_ct2 = np.std(aucs_ct2)
    aucs_ct_perms_shuff.append(aucs_ct2)
    mean_auc_ct_perms_shuff.append(mean_auc_ct2)
    
    diff_mean_auc_z_ct_shuff = mean_auc_z2 - mean_auc_ct2
    diff_mean_auc_z_ct_perms_shuff.append(diff_mean_auc_z_ct_shuff)
    plt.close()
    
auc_z_shuff = np.array(aucs_z_perms_shuff)
mean_auc_z_shuff = np.array(mean_auc_z_perms_shuff)
auc_ct_shuff = np.array(aucs_ct_perms_shuff)
mean_auc_ct_shuff = np.array(mean_auc_ct_perms_shuff)
np.save('/sz_svm_auc_ct_shuff.npy', auc_z_shuff)
np.save('sz_svm_mean_auc_ct_shuff.npy', mean_auc_z_shuff)
np.save('sz_svm_auc_ct_shuff.npy', auc_ct_shuff)
np.save('sz_svm_mean_auc_ct_shuff.npy', mean_auc_ct_shuff)

diff_shuff = np.array(diff_mean_auc_z_ct_perms_shuff)
np.save('sz_svm_diff_shuff_ct.npy', diff_shuff)
