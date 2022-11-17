import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, linalg
from sklearn import preprocessing, decomposition, linear_model, metrics 
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


# Cortical Thickness
hcp_ct_z = np.load('hcp_ct_z.npy')
hcp_ct_resid = np.load('hcp_ct_resid.npy')
hcp_ct_g = np.load('hcp_ct_g.npy')


# generate train/test splits
np.random.seed(42)
n_train = int(0.9 * hcp_ct_z.shape[0])

train_idxs = np.random.choice(range(hcp_ct_z.shape[0]), size=n_train, replace=False)
test_idxs = np.array([x for x in range(hcp_ct_z.shape[0]) if x not in train_idxs])

train_data_z = hcp_ct_z[train_idxs, :]
test_data_z = hcp_ct_z[test_idxs, :]

train_data_ct = hcp_ct_resid[train_idxs, :]
test_data_ct = hcp_ct_resid[test_idxs, :]

train_phen = hcp_ct_g[train_idxs]
test_phen = hcp_ct_g[test_idxs]

bbs_r2_perm = []
bbs_mse_perm = []
bbs_r2_perm_shuff = []
bbs_mse_perm_shuff = []
bbs_test_r2_z_array = []
bbs_test_mse_z_array = []
bbs_test_r2_ct_array = []
bbs_test_mse_ct_array = []
bbs_test_r2_z_shuff_array = []
bbs_test_mse_z_shuff_array = []
bbs_test_r2_ct_shuff_array = []
bbs_test_mse_ct_shuff_array = []

for perm in range(10000):
    random_state_perm = np.random.RandomState(perm)
    hcp_g_shuff = shuffle(hcp_ct_g, random_state=random_state_perm)
    train_phen_shuff = hcp_g_shuff[train_idxs]
    test_phen_shuff = hcp_g_shuff[test_idxs]

    # mean center train/test data (using train means)
    train_mu_centered_z = (train_data_z - train_data_z.mean(axis=0))
    test_mu_centered_z = (test_data_z - train_data_z.mean(axis=0))

    train_mu_centered_ct = (train_data_ct - train_data_ct.mean(axis=0))
    test_mu_centered_ct = (test_data_ct - train_data_ct.mean(axis=0))

    # from pca documentation, "the input data is centered but not scaled for each feature before applying the SVD"
    pca_model_z = decomposition.PCA(n_components=15).fit(train_data_z)
    pca_model_ct = decomposition.PCA(n_components=15).fit(train_data_ct)

    train_transformed_z = pca_model_z.transform(train_data_z)
    test_transformed_z = pca_model_z.transform(test_data_z)
    train_transformed_ct = pca_model_ct.transform(train_data_ct)
    test_transformed_ct = pca_model_ct.transform(test_data_ct)

    # OLS using sklearn
    lr_model_z = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    lr_model_z.fit(train_transformed_z, train_phen)
    train_pred_phen_lr_model_z = lr_model_z.predict(train_transformed_z)
    test_pred_phen_lr_model_z = lr_model_z.predict(test_transformed_z)

    lr_model_z_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    lr_model_z_shuff.fit(train_transformed_z, train_phen_shuff)
    train_pred_phen_lr_model_z_shuff = lr_model_z_shuff.predict(train_transformed_z)
    test_pred_phen_lr_model_z_shuff = lr_model_z_shuff.predict(test_transformed_z)

    # OLS using sklearn
    lr_model_ct = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    lr_model_ct.fit(train_transformed_ct, train_phen)
    train_pred_phen_lr_model_ct = lr_model_ct.predict(train_transformed_ct)
    test_pred_phen_lr_model_ct = lr_model_ct.predict(test_transformed_ct)

    lr_model_ct_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    lr_model_ct_shuff.fit(train_transformed_ct, train_phen_shuff)
    train_pred_phen_lr_model_ct_shuff = lr_model_ct_shuff.predict(train_transformed_ct)
    test_pred_phen_lr_model_ct_shuff = lr_model_ct_shuff.predict(test_transformed_ct)

    # HCP Accuracy of Predictions (deviations)
    train_r2_z = metrics.r2_score(train_phen, train_pred_phen_lr_model_z)
    test_r2_z = metrics.r2_score(test_phen, test_pred_phen_lr_model_z)
    train_mse_z = metrics.mean_squared_error(train_phen, train_pred_phen_lr_model_z)
    test_mse_z = metrics.mean_squared_error(test_phen, test_pred_phen_lr_model_z)

    # HCP Accuracy of Predictions (deviations - shuffled)
    train_r2_z_shuff = metrics.r2_score(train_phen_shuff, train_pred_phen_lr_model_z_shuff)
    test_r2_z_shuff = metrics.r2_score(test_phen_shuff, test_pred_phen_lr_model_z_shuff)
    train_mse_z_shuff = metrics.mean_squared_error(train_phen_shuff, train_pred_phen_lr_model_z_shuff)
    test_mse_z_shuff = metrics.mean_squared_error(test_phen_shuff, test_pred_phen_lr_model_z_shuff)

    # HCP Accuracy of Predictions (cortical thickness)
    train_r2_ct = metrics.r2_score(train_phen, train_pred_phen_lr_model_ct)
    test_r2_ct = metrics.r2_score(test_phen, test_pred_phen_lr_model_ct)
    train_mse_ct = metrics.mean_squared_error(train_phen, train_pred_phen_lr_model_ct)
    test_mse_ct = metrics.mean_squared_error(test_phen, test_pred_phen_lr_model_ct)

    # HCP Accuracy of Predictions (cortical thickness - shuffled)
    train_r2_ct_shuff = metrics.r2_score(train_phen_shuff, train_pred_phen_lr_model_ct_shuff)
    test_r2_ct_shuff = metrics.r2_score(test_phen_shuff, test_pred_phen_lr_model_ct_shuff)
    train_mse_ct_shuff = metrics.mean_squared_error(train_phen_shuff, train_pred_phen_lr_model_ct_shuff)
    test_mse_ct_shuff = metrics.mean_squared_error(test_phen_shuff, test_pred_phen_lr_model_ct_shuff)

    bbs_test_r2_z_array.append(test_r2_z)
    bbs_test_mse_z_array.append(test_mse_z)
    bbs_test_r2_z_shuff_array.append(test_r2_z_shuff)
    bbs_test_mse_z_shuff_array.append(test_mse_z_shuff)
    bbs_test_r2_ct_array.append(test_r2_ct)
    bbs_test_mse_ct_array.append(test_mse_ct)
    bbs_test_r2_ct_shuff_array.append(test_r2_ct_shuff)
    bbs_test_mse_ct_shuff_array.append(test_mse_ct_shuff)

    diff_test_r = test_r2_z - test_r2_ct
    diff_test_mse = test_mse_z - test_mse_ct
    diff_test_r_shuff = test_r2_z_shuff - test_r2_ct_shuff
    diff_test_mse_shuff = test_mse_z_shuff - test_mse_ct_shuff

    bbs_mse_perm.append(diff_test_mse)
    bbs_r2_perm.append(diff_test_r)
    bbs_mse_perm_shuff.append(diff_test_mse_shuff)
    bbs_r2_perm_shuff.append(diff_test_r_shuff)

bbs_test_r2_z_arr = np.array(bbs_test_r2_z_array)
bbs_test_mse_z_arr = np.array(bbs_test_mse_z_array)
bbs_test_r2_z_shuff_arr = np.array(bbs_test_r2_z_shuff_array)
bbs_test_mse_z_shuff_arr = np.array(bbs_test_mse_z_shuff_array)
bbs_test_r2_ct_arr = np.array(bbs_test_r2_ct_array)
bbs_test_mse_ct_arr = np.array(bbs_test_mse_ct_array)
bbs_test_r2_ct_shuff_arr = np.array(bbs_test_r2_ct_shuff_array)
bbs_test_mse_ct_shuff_arr = np.array(bbs_test_mse_ct_shuff_array)
np.save('hcp_ct_bbs_test_r2_z_arr.npy', bbs_test_r2_z_arr)
np.save('hcp_ct_bbs_test_mse_z_arr.npy', bbs_test_mse_z_arr)
np.save('hcp_ct_bbs_test_r2_z_shuff_arr.npy', bbs_test_r2_z_shuff_arr)
np.save('hcp_ct_bbs_test_mse_z_shuff_arr.npy', bbs_test_mse_z_shuff_arr)
np.save('hcp_ct_bbs_test_r2_ct_arr.npy', bbs_test_r2_ct_arr)
np.save('hcp_ct_bbs_test_mse_ct_arr.npy', bbs_test_mse_ct_arr)
np.save('hcp_ct_bbs_test_r2_ct_shuff_arr.npy', bbs_test_r2_ct_shuff_arr)
np.save('hcp_ct_test_mse_ct_shuff_arr.npy', bbs_test_mse_ct_shuff_arr)

bbs_r2_perm_arr = np.array(bbs_r2_perm)
bbs_r2_perm_shuff_arr = np.array(bbs_r2_perm_shuff)
bbs_mse_perm_arr = np.array(bbs_mse_perm)
bbs_mse_perm_shuff_arr = np.array(bbs_mse_perm_shuff)
np.save('hcp_ct_bbs_r2_diff_perm.npy',bbs_r2_perm_arr)
np.save('hcp_ct_bbs_mse_diff_perm.npy',bbs_mse_perm_arr)
np.save('hcp_ct_bbs_r2_diff_perm_shuff.npy',bbs_r2_perm_shuff_arr)
np.save('hcp_ct_bbs_mse_diff_perm_shuff.npy',bbs_mse_perm_shuff_arr)