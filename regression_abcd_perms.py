import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, linalg
from sklearn import preprocessing, decomposition, linear_model, metrics 
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

abcd_z = np.load('abcd_z.npy')
abcd_ct = np.load('abcd_ct.npy')
abcd_g = np.load('abcd_g.npy')

# generate train/test splits
np.random.seed(42)
n_train = int(0.8 * abcd_z.shape[0])

train_idxs = np.random.choice(range(abcd_z.shape[0]), size=n_train, replace=False)
test_idxs = np.array([x for x in range(abcd_z.shape[0]) if x not in train_idxs])

train_data_z = abcd_z[train_idxs, :]
test_data_z = abcd_z[test_idxs, :]

train_data_ct = abcd_ct[train_idxs, :]
test_data_ct = abcd_ct[test_idxs, :]

train_phen = abcd_g[train_idxs]
test_phen = abcd_g[test_idxs]

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

cpm_perm = []
cpm_perm_shuff = []
cpm_test_mse_z_array = []
cpm_test_mse_ct_array = []
cpm_test_mse_z_shuff_array = []
cpm_test_mse_ct_shuff_array = []

lasso_perm = []
lasso_perm_shuff = []
lasso_test_mse_z_array = []
lasso_test_mse_ct_array = []
lasso_test_mse_z_shuff_array = []
lasso_test_mse_ct_shuff_array = []

ridge_perm = []
ridge_perm_shuff = []
ridge_test_mse_z_array = []
ridge_test_mse_ct_array = []
ridge_test_mse_z_shuff_array = []
ridge_test_mse_ct_shuff_array = []

enet_perm = []
enet_perm_shuff = []
enet_test_mse_z_array = []
enet_test_mse_ct_array = []
enet_test_mse_z_shuff_array = []
enet_test_mse_ct_shuff_array = []

for perm in range(10000):
    random_state_perm = np.random.RandomState(perm)
    abcd_g_shuff = shuffle(abcd_g, random_state=random_state_perm)
    train_phen_shuff = abcd_g_shuff[train_idxs]
    test_phen_shuff = abcd_g_shuff[test_idxs]

    # mean center train/test data (using train means)
    train_mu_centered_z = (train_data_z - train_data_z.mean(axis=0))
    test_mu_centered_z = (test_data_z - train_data_z.mean(axis=0))

    train_mu_centered_ct = (train_data_ct - train_data_ct.mean(axis=0))
    test_mu_centered_ct = (test_data_ct - train_data_ct.mean(axis=0))

    # from pca documentation, "the input data is centered but not scaled for each feature before applying the SVD"
    pca_model_z = decomposition.PCA(n_components=75).fit(train_data_z)
    pca_model_ct = decomposition.PCA(n_components=75).fit(train_data_ct)

    train_transformed_z = pca_model_z.transform(train_data_z)
    test_transformed_z = pca_model_z.transform(test_data_z)
    train_transformed_ct = pca_model_ct.transform(train_data_ct)
    test_transformed_ct = pca_model_ct.transform(test_data_ct)

    # fast OLS using matrix math
    # we will check that this matches sklearn results later

    # fit ols model on dimension reduced train data
    train_features_z = np.hstack([np.ones((train_transformed_z.shape[0], 1)), train_transformed_z])
    train_features_inv_z = linalg.pinv2(train_features_z)
    train_betas_z = np.dot(train_features_inv_z, train_phen)
    train_pred_phen_z = np.dot(train_features_z, train_betas_z)
    train_betas_z_shuff = np.dot(train_features_inv_z, train_phen_shuff)
    train_pred_phen_z_shuff = np.dot(train_features_z, train_betas_z_shuff)

    # fit ols model on dimension reduced test data
    test_features_z = np.hstack([np.ones((test_transformed_z.shape[0], 1)), test_transformed_z])
    test_pred_phen_z = np.dot(test_features_z, train_betas_z)
    test_pred_phen_z_shuff = np.dot(test_features_z, train_betas_z_shuff)

    # fast OLS using matrix math
    # we will check that this matches sklearn results later

    # fit ols model on dimension reduced train data
    train_features_ct = np.hstack([np.ones((train_transformed_ct.shape[0], 1)), train_transformed_ct])
    train_features_inv_ct = linalg.pinv2(train_features_ct)
    train_betas_ct = np.dot(train_features_inv_ct, train_phen)
    train_pred_phen_ct = np.dot(train_features_ct, train_betas_ct)
    train_betas_ct_shuff = np.dot(train_features_inv_ct, train_phen_shuff)
    train_pred_phen_ct_shuff = np.dot(train_features_ct, train_betas_ct_shuff)

    # fit ols model on dimension reduced test data
    test_features_ct = np.hstack([np.ones((test_transformed_ct.shape[0], 1)), test_transformed_ct])
    test_pred_phen_ct = np.dot(test_features_ct, train_betas_ct)
    test_pred_phen_ct_shuff = np.dot(test_features_ct, train_betas_ct_shuff)

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

    # ensure matrix math predictions and sklearn predictions are accurate to 5 decimals
    assert np.allclose(np.round(train_pred_phen_z - train_pred_phen_lr_model_z, 5), 0), 'Failed'
    assert np.allclose(np.round(test_pred_phen_z - test_pred_phen_lr_model_z, 5), 0), 'Failed'
    assert np.allclose(np.round(train_pred_phen_z_shuff - train_pred_phen_lr_model_z_shuff, 5), 0), 'Failed'
    assert np.allclose(np.round(test_pred_phen_z_shuff - test_pred_phen_lr_model_z_shuff, 5), 0), 'Failed'
    print('Passed')

    # ensure matrix math predictions and sklearn predictions are accurate to 5 decimals
    assert np.allclose(np.round(train_pred_phen_ct - train_pred_phen_lr_model_ct, 5), 0), 'Failed'
    assert np.allclose(np.round(test_pred_phen_ct - test_pred_phen_lr_model_ct, 5), 0), 'Failed'
    assert np.allclose(np.round(train_pred_phen_ct_shuff - train_pred_phen_lr_model_ct_shuff, 5), 0), 'Failed'
    assert np.allclose(np.round(test_pred_phen_ct_shuff - test_pred_phen_lr_model_ct_shuff, 5), 0), 'Failed'
    print('Passed')

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

    # Connectome Predictive Modeling
    # correlation train_brain with train_phenotype
    train_z_pheno_corr_p = [stats.pearsonr(train_data_z[:, i], train_phen) for i in range(train_data_z.shape[1])]  # train_pheno_corr_p: (259200, )
    train_ct_pheno_corr_p = [stats.pearsonr(train_data_ct[:, i], train_phen) for i in range(train_data_ct.shape[1])]  # train_pheno_corr_p: (259200, )
    train_z_pheno_corr_p_shuff = [stats.pearsonr(train_data_z[:, i], train_phen_shuff) for i in range(train_data_z.shape[1])]  # train_pheno_corr_p: (259200, )
    train_ct_pheno_corr_p_shuff = [stats.pearsonr(train_data_ct[:, i], train_phen_shuff) for i in range(train_data_ct.shape[1])]  # train_pheno_corr_p: (259200, )

    # split into positive and negative correlations 
    # and keep edges with p values below threshold
    pval_threshold = 0.01

    train_z_corrs = np.array([x[0] for x in train_z_pheno_corr_p])
    train_z_pvals = np.array([x[1] for x in train_z_pheno_corr_p])
    train_z_corrs_shuff = np.array([x[0] for x in train_z_pheno_corr_p_shuff])
    train_z_pvals_shuff = np.array([x[1] for x in train_z_pheno_corr_p_shuff])
    keep_edges_pos_z = (train_z_corrs > 0) & (train_z_pvals < pval_threshold)
    keep_edges_neg_z = (train_z_corrs < 0) & (train_z_pvals < pval_threshold)
    keep_edges_pos_z_shuff = (train_z_corrs_shuff > 0) & (train_z_pvals_shuff < pval_threshold)
    keep_edges_neg_z_shuff = (train_z_corrs_shuff < 0) & (train_z_pvals_shuff < pval_threshold)

    train_ct_corrs = np.array([x[0] for x in train_ct_pheno_corr_p])
    train_ct_pvals = np.array([x[1] for x in train_ct_pheno_corr_p])
    train_ct_corrs_shuff = np.array([x[0] for x in train_ct_pheno_corr_p_shuff])
    train_ct_pvals_shuff = np.array([x[1] for x in train_ct_pheno_corr_p_shuff])
    keep_edges_pos_ct = (train_ct_corrs > 0) & (train_ct_pvals < pval_threshold)
    keep_edges_neg_ct = (train_ct_corrs < 0) & (train_ct_pvals < pval_threshold)
    keep_edges_pos_ct_shuff = (train_ct_corrs_shuff > 0) & (train_ct_pvals_shuff < pval_threshold)
    keep_edges_neg_ct_shuff = (train_ct_corrs_shuff < 0) & (train_ct_pvals_shuff < pval_threshold)

    train_pos_edges_sum_z = train_data_z[:, keep_edges_pos_z].sum(1)
    train_neg_edges_sum_z = train_data_z[:, keep_edges_neg_z].sum(1)
    train_pos_edges_sum_ct = train_data_ct[:, keep_edges_pos_ct].sum(1)
    train_neg_edges_sum_ct = train_data_ct[:, keep_edges_neg_ct].sum(1)

    train_pos_edges_sum_z_shuff = train_data_z[:, keep_edges_pos_z_shuff].sum(1)
    train_neg_edges_sum_z_shuff = train_data_z[:, keep_edges_neg_z_shuff].sum(1)
    train_pos_edges_sum_ct_shuff = train_data_ct[:, keep_edges_pos_ct_shuff].sum(1)
    train_neg_edges_sum_ct_shuff = train_data_ct[:, keep_edges_neg_ct_shuff].sum(1)

    fit_pos_z = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_pos_edges_sum_z.reshape(-1, 1), train_phen)
    fit_neg_z = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_neg_edges_sum_z.reshape(-1, 1), train_phen)
    fit_pos_ct = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_pos_edges_sum_ct.reshape(-1, 1), train_phen)
    fit_neg_ct = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_neg_edges_sum_ct.reshape(-1, 1), train_phen)

    fit_pos_z_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_pos_edges_sum_z_shuff.reshape(-1, 1), train_phen_shuff)
    fit_neg_z_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_neg_edges_sum_z_shuff.reshape(-1, 1), train_phen_shuff)
    fit_pos_ct_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_pos_edges_sum_ct_shuff.reshape(-1, 1), train_phen_shuff)
    fit_neg_ct_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(train_neg_edges_sum_ct_shuff.reshape(-1, 1), train_phen_shuff)

    # combine positive/negative edges in one linear regression model
    fit_pos_neg_z = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(np.stack((train_pos_edges_sum_z, train_neg_edges_sum_z)).T, train_phen)
    fit_pos_neg_ct = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(np.stack((train_pos_edges_sum_ct, train_neg_edges_sum_ct)).T, train_phen)
    fit_pos_neg_z_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(np.stack((train_pos_edges_sum_z_shuff, train_neg_edges_sum_z_shuff)).T, train_phen_shuff)
    fit_pos_neg_ct_shuff = linear_model.LinearRegression(fit_intercept=True, normalize=False).fit(np.stack((train_pos_edges_sum_ct_shuff, train_neg_edges_sum_ct_shuff)).T, train_phen_shuff)

    # evaluate out of sample performance 
    test_pos_edges_sum_z = test_data_z[:, keep_edges_pos_z].sum(1)
    test_neg_edges_sum_z = test_data_z[:, keep_edges_neg_z].sum(1)
    test_pos_edges_sum_z_shuff = test_data_z[:, keep_edges_pos_z_shuff].sum(1)
    test_neg_edges_sum_z_shuff = test_data_z[:, keep_edges_neg_z_shuff].sum(1)

    pos_test_error_z = metrics.mean_squared_error(test_phen, fit_pos_z.predict(test_pos_edges_sum_z.reshape(-1, 1)))
    neg_test_error_z = metrics.mean_squared_error(test_phen, fit_neg_z.predict(test_neg_edges_sum_z.reshape(-1, 1)))
    pos_neg_test_error_z = metrics.mean_squared_error(test_phen, fit_pos_neg_z.predict(np.stack((test_pos_edges_sum_z, test_neg_edges_sum_z)).T))
    pos_test_error_z_shuff = metrics.mean_squared_error(test_phen_shuff, fit_pos_z_shuff.predict(test_pos_edges_sum_z_shuff.reshape(-1, 1)))
    neg_test_error_z_shuff = metrics.mean_squared_error(test_phen_shuff, fit_neg_z_shuff.predict(test_neg_edges_sum_z_shuff.reshape(-1, 1)))
    pos_neg_test_error_z_shuff = metrics.mean_squared_error(test_phen_shuff, fit_pos_neg_z_shuff.predict(np.stack((test_pos_edges_sum_z_shuff, test_neg_edges_sum_z_shuff)).T))

    test_pos_edges_sum_ct = test_data_ct[:, keep_edges_pos_ct].sum(1)
    test_neg_edges_sum_ct = test_data_ct[:, keep_edges_neg_ct].sum(1)
    test_pos_edges_sum_ct_shuff = test_data_ct[:, keep_edges_pos_ct_shuff].sum(1)
    test_neg_edges_sum_ct_shuff = test_data_ct[:, keep_edges_neg_ct_shuff].sum(1)

    pos_test_error_ct = metrics.mean_squared_error(test_phen, fit_pos_ct.predict(test_pos_edges_sum_ct.reshape(-1, 1)))
    neg_test_error_ct = metrics.mean_squared_error(test_phen, fit_neg_ct.predict(test_neg_edges_sum_ct.reshape(-1, 1)))
    pos_neg_test_error_ct = metrics.mean_squared_error(test_phen, fit_pos_neg_ct.predict(np.stack((test_pos_edges_sum_ct, test_neg_edges_sum_ct)).T))
    pos_test_error_ct_shuff = metrics.mean_squared_error(test_phen_shuff, fit_pos_ct_shuff.predict(test_pos_edges_sum_ct_shuff.reshape(-1, 1)))
    neg_test_error_ct_shuff = metrics.mean_squared_error(test_phen_shuff, fit_neg_ct_shuff.predict(test_neg_edges_sum_ct_shuff.reshape(-1, 1)))
    pos_neg_test_error_ct_shuff = metrics.mean_squared_error(test_phen_shuff, fit_pos_neg_ct_shuff.predict(np.stack((test_pos_edges_sum_ct_shuff, test_neg_edges_sum_ct_shuff)).T))

    cpm_test_mse_z_array.append(pos_neg_test_error_z)
    cpm_test_mse_ct_array.append(pos_neg_test_error_ct)
    cpm_test_mse_z_shuff_array.append(pos_neg_test_error_z_shuff)
    cpm_test_mse_ct_shuff_array.append(pos_neg_test_error_ct_shuff)

    diff_test_mse = pos_neg_test_error_z - pos_neg_test_error_ct
    diff_test_mse_shuff = pos_neg_test_error_z_shuff - pos_neg_test_error_ct_shuff
    
    cpm_perm.append(diff_test_mse)
    cpm_perm_shuff.append(diff_test_mse_shuff)

    # Lasso
    alpha_grid = np.array([10**a for a in np.arange(-3, 3, 0.25)])

    lassoCV_model_z = linear_model.LassoCV(cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, fit_intercept=True, normalize=False, random_state=42, verbose=True, n_jobs=5).fit(train_data_z, train_phen)
    lassoCV_model_ct = linear_model.LassoCV(cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, fit_intercept=True, normalize=False, random_state=42, verbose=True, n_jobs=5).fit(train_data_ct, train_phen)
    lassoCV_model_z_shuff = linear_model.LassoCV(cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, fit_intercept=True, normalize=False, random_state=42, verbose=True, n_jobs=5).fit(train_data_z, train_phen_shuff)
    lassoCV_model_ct_shuff = linear_model.LassoCV(cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, fit_intercept=True, normalize=False, random_state=42, verbose=True, n_jobs=5).fit(train_data_ct, train_phen_shuff)
    
    # based on cv results above, set alpha=100
    lasso_model_z = linear_model.Lasso(alpha=lassoCV_model_z.alpha_, fit_intercept=True, normalize=False).fit(train_data_z, train_phen)
    lasso_model_ct = linear_model.Lasso(alpha=lassoCV_model_ct.alpha_, fit_intercept=True, normalize=False).fit(train_data_ct, train_phen)
    lasso_model_z_shuff = linear_model.Lasso(alpha=lassoCV_model_z.alpha_, fit_intercept=True, normalize=False).fit(train_data_z, train_phen_shuff)
    lasso_model_ct_shuff = linear_model.Lasso(alpha=lassoCV_model_ct.alpha_, fit_intercept=True, normalize=False).fit(train_data_ct, train_phen_shuff)

    train_preds_lasso_model_z = lasso_model_z.predict(train_data_z)
    test_preds_lasso_model_z = lasso_model_z.predict(test_data_z)
    train_preds_lasso_model_z_shuff = lasso_model_z_shuff.predict(train_data_z)
    test_preds_lasso_model_z_shuff = lasso_model_z_shuff.predict(test_data_z)

    train_mse_z = metrics.mean_squared_error(train_phen, train_preds_lasso_model_z)
    test_mse_z = metrics.mean_squared_error(test_phen, test_preds_lasso_model_z)
    train_mse_z_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_lasso_model_z_shuff)
    test_mse_z_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_lasso_model_z_shuff)

    train_preds_lasso_model_ct = lasso_model_ct.predict(train_data_ct)
    test_preds_lasso_model_ct = lasso_model_ct.predict(test_data_ct)
    train_preds_lasso_model_ct_shuff = lasso_model_ct_shuff.predict(train_data_ct)
    test_preds_lasso_model_ct_shuff = lasso_model_ct_shuff.predict(test_data_ct)

    train_mse_ct = metrics.mean_squared_error(train_phen, train_preds_lasso_model_ct)
    test_mse_ct = metrics.mean_squared_error(test_phen, test_preds_lasso_model_ct)
    train_mse_ct_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_lasso_model_ct_shuff)
    test_mse_ct_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_lasso_model_ct_shuff)

    lasso_test_mse_z_array.append(test_mse_z)
    lasso_test_mse_ct_array.append(test_mse_ct)
    lasso_test_mse_z_shuff_array.append(test_mse_z_shuff)
    lasso_test_mse_ct_shuff_array.append(test_mse_ct_shuff)

    diff_train_mse = train_mse_z - train_mse_ct
    diff_train_mse_shuff = train_mse_z_shuff - train_mse_ct_shuff

    lasso_perm.append(diff_train_mse)
    lasso_perm_shuff.append(diff_train_mse_shuff)

    # Ridge
    with warnings.catch_warnings():
        # ignore matrix decomposition errors
        warnings.simplefilter("ignore")
        ridgeCV_model_z = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, cv=5).fit(train_data_z, train_phen)
        ridgeCV_model_ct = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, cv=5).fit(train_data_ct, train_phen)
        ridgeCV_model_z_shuff = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, cv=5).fit(train_data_z, train_phen_shuff)
        ridgeCV_model_ct_shuff = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, cv=5).fit(train_data_ct, train_phen_shuff)
    
    ridge_alpha_z = ridgeCV_model_z.alpha_
    ridge_alpha_ct = ridgeCV_model_ct.alpha_
    ridge_alpha_z_shuff = ridgeCV_model_z_shuff.alpha_
    ridge_alpha_ct_shuff = ridgeCV_model_ct_shuff.alpha_

    ridge_model_z = linear_model.Ridge(alpha=ridge_alpha_z, fit_intercept=True, normalize=False).fit(train_data_z, train_phen)
    ridge_model_ct = linear_model.Ridge(alpha=ridge_alpha_ct, fit_intercept=True, normalize=False).fit(train_data_ct, train_phen)
    ridge_model_z_shuff = linear_model.Ridge(alpha=ridge_alpha_z_shuff, fit_intercept=True, normalize=False).fit(train_data_z, train_phen_shuff)
    ridge_model_ct_shuff = linear_model.Ridge(alpha=ridge_alpha_ct_shuff, fit_intercept=True, normalize=False).fit(train_data_ct, train_phen_shuff)

    train_preds_ridge_model_z = ridge_model_z.predict(train_data_z)
    test_preds_ridge_model_z = ridge_model_z.predict(test_data_z)
    train_preds_ridge_model_z_shuff = ridge_model_z_shuff.predict(train_data_z)
    test_preds_ridge_model_z_shuff = ridge_model_z_shuff.predict(test_data_z)

    train_mse_z = metrics.mean_squared_error(train_phen, train_preds_ridge_model_z)
    test_mse_z = metrics.mean_squared_error(test_phen, test_preds_ridge_model_z)
    train_mse_z_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_ridge_model_z_shuff)
    test_mse_z_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_ridge_model_z_shuff)

    train_preds_ridge_model_ct = ridge_model_ct.predict(train_data_ct)
    test_preds_ridge_model_ct = ridge_model_ct.predict(test_data_ct)
    train_preds_ridge_model_ct_shuff = ridge_model_ct_shuff.predict(train_data_ct)
    test_preds_ridge_model_ct_shuff = ridge_model_ct_shuff.predict(test_data_ct)

    train_mse_ct = metrics.mean_squared_error(train_phen, train_preds_ridge_model_ct)
    test_mse_ct = metrics.mean_squared_error(test_phen, test_preds_ridge_model_ct)
    train_mse_ct_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_ridge_model_ct_shuff)
    test_mse_ct_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_ridge_model_ct_shuff)

    ridge_test_mse_z_array.append(test_mse_z)
    ridge_test_mse_ct_array.append(test_mse_ct)
    ridge_test_mse_z_shuff_array.append(test_mse_z_shuff)
    ridge_test_mse_ct_shuff_array.append(test_mse_ct_shuff)

    diff_test_mse = test_mse_z - test_mse_ct
    diff_test_mse_shuff = test_mse_z_shuff - test_mse_ct_shuff

    ridge_perm.append(diff_test_mse)
    ridge_perm_shuff.append(diff_test_mse_shuff)

    # Elastic Net
    elasticnetCV_model_z = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, random_state=42, verbose=True, n_jobs=5).fit(train_data_z, train_phen)
    elasticnetCV_model_ct = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, random_state=42, verbose=True, n_jobs=5).fit(train_data_ct, train_phen)
    elasticnetCV_model_z_shuff = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, random_state=42, verbose=True, n_jobs=5).fit(train_data_z, train_phen_shuff)
    elasticnetCV_model_ct_shuff = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_alphas=len(alpha_grid), alphas=alpha_grid, random_state=42, verbose=True, n_jobs=5).fit(train_data_ct, train_phen_shuff)

    elasticnet_model_z = linear_model.ElasticNet(alpha=elasticnetCV_model_z.alpha_, l1_ratio=elasticnetCV_model_z.l1_ratio_, fit_intercept=True, normalize=False, random_state=42).fit(train_data_z, train_phen)
    elasticnet_model_ct = linear_model.ElasticNet(alpha=elasticnetCV_model_ct.alpha_, l1_ratio=elasticnetCV_model_ct.l1_ratio_, fit_intercept=True, normalize=False, random_state=42).fit(train_data_ct, train_phen)
    elasticnet_model_z_shuff = linear_model.ElasticNet(alpha=elasticnetCV_model_z_shuff.alpha_, l1_ratio=elasticnetCV_model_z_shuff.l1_ratio_, fit_intercept=True, normalize=False, random_state=42).fit(train_data_z, train_phen_shuff)
    elasticnet_model_ct_shuff = linear_model.ElasticNet(alpha=elasticnetCV_model_ct_shuff.alpha_, l1_ratio=elasticnetCV_model_ct_shuff.l1_ratio_, fit_intercept=True, normalize=False, random_state=42).fit(train_data_ct, train_phen_shuff)

    train_preds_en_model_z = elasticnet_model_z.predict(train_data_z)
    test_preds_en_model_z = elasticnet_model_z.predict(test_data_z)
    train_preds_en_model_ct = elasticnet_model_ct.predict(train_data_ct)
    test_preds_en_model_ct = elasticnet_model_ct.predict(test_data_ct)
    train_preds_en_model_z_shuff = elasticnet_model_z_shuff.predict(train_data_z)
    test_preds_en_model_z_shuff = elasticnet_model_z_shuff.predict(test_data_z)
    train_preds_en_model_ct_shuff = elasticnet_model_ct_shuff.predict(train_data_ct)
    test_preds_en_model_ct_shuff = elasticnet_model_ct_shuff.predict(test_data_ct)

    train_mse_z = metrics.mean_squared_error(train_phen, train_preds_en_model_z)
    test_mse_z = metrics.mean_squared_error(test_phen, test_preds_en_model_z)
    train_mse_ct = metrics.mean_squared_error(train_phen, train_preds_en_model_ct)
    test_mse_ct = metrics.mean_squared_error(test_phen, test_preds_en_model_ct)
    train_mse_z_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_en_model_z_shuff)
    test_mse_z_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_en_model_z_shuff)
    train_mse_ct_shuff = metrics.mean_squared_error(train_phen_shuff, train_preds_en_model_ct_shuff)
    test_mse_ct_shuff = metrics.mean_squared_error(test_phen_shuff, test_preds_en_model_ct_shuff)

    enet_test_mse_z_array.append(test_mse_z)
    enet_test_mse_ct_array.append(test_mse_ct)
    enet_test_mse_z_shuff_array.append(test_mse_z_shuff)
    enet_test_mse_ct_shuff_array.append(test_mse_ct_shuff)

    diff_test_mse_z = test_mse_z - test_mse_ct
    diff_test_mse_z_shuff = test_mse_z_shuff - test_mse_ct_shuff

    enet_perm.append(diff_test_mse_z)
    enet_perm_shuff.append(diff_test_mse_z_shuff)

bbs_test_r2_z_arr = np.array(bbs_test_r2_z_array)
bbs_test_mse_z_arr = np.array(bbs_test_mse_z_array)
bbs_test_r2_z_shuff_arr = np.array(bbs_test_r2_z_shuff_array)
bbs_test_mse_z_shuff_arr = np.array(bbs_test_mse_z_shuff_array)
bbs_test_r2_ct_arr = np.array(bbs_test_r2_ct_array)
bbs_test_mse_ct_arr = np.array(bbs_test_mse_ct_array)
bbs_test_r2_ct_shuff_arr = np.array(bbs_test_r2_ct_shuff_array)
bbs_test_mse_ct_shuff_arr = np.array(bbs_test_mse_ct_shuff_array)
np.save('abcd_bbs_test_r2_z_arr.npy', bbs_test_r2_z_arr)
np.save('abcd_bbs_test_mse_z_arr.npy', bbs_test_mse_z_arr)
np.save('abcd_bbs_test_r2_z_shuff_arr.npy', bbs_test_r2_z_shuff_arr)
np.save('abcd_bbs_test_mse_z_shuff_arr.npy', bbs_test_mse_z_shuff_arr)
np.save('abcd_bbs_test_r2_ct_arr.npy', bbs_test_r2_ct_arr)
np.save('abcd_bbs_test_mse_ct_arr.npy', bbs_test_mse_ct_arr)
np.save('abcd_bbs_test_r2_ct_shuff_arr.npy', bbs_test_r2_ct_shuff_arr)
np.save('abcd_bbs_test_mse_ct_shuff_arr.npy', bbs_test_mse_ct_shuff_arr)

bbs_r2_perm_arr = np.array(bbs_r2_perm)
bbs_r2_perm_shuff_arr = np.array(bbs_r2_perm_shuff)
bbs_mse_perm_arr = np.array(bbs_mse_perm)
bbs_mse_perm_shuff_arr = np.array(bbs_mse_perm_shuff)
np.save('abcd_bbs_r2_diff_perm.npy',bbs_r2_perm_arr)
np.save('abcd_bbs_mse_diff_perm.npy',bbs_mse_perm_arr)
np.save('abcd_bbs_r2_diff_perm_shuff.npy',bbs_r2_perm_shuff_arr)
np.save('abcd_bbs_mse_diff_perm_shuff.npy',bbs_mse_perm_shuff_arr)

cpm_test_mse_z_arr = np.array(cpm_test_mse_z_array)
cpm_test_mse_ct_arr = np.array(cpm_test_mse_ct_array)
cpm_test_mse_z_shuff = np.array(cpm_test_mse_z_shuff_array)
cpm_test_mse_ct_shuff_arr = np.array(cpm_test_mse_ct_shuff_array)
np.save('abcd_cpm_test_mse_z_arr.npy', cpm_test_mse_z_arr)
np.save('abcd_cpm_test_mse_ct_arr.npy', cpm_test_mse_ct_arr)
np.save('abcd_cpm_test_mse_z_shuff_arr.npy', cpm_test_mse_z_shuff)
np.save('abcd_cpm_test_mse_ct_shuff_arr.npy', cpm_test_mse_ct_shuff_arr)

cpm_perm_arr = np.array(cpm_perm)
cpm_perm_shuff_arr = np.array(cpm_perm_shuff)
np.save('abcd_cpm_perm.npy',cpm_perm_arr)
np.save('abcd_cpm_perm_shuff.npy',cpm_perm_shuff_arr)

lasso_test_mse_z_arr = np.array(lasso_test_mse_z_array)
lasso_test_mse_ct_arr = np.array(lasso_test_mse_ct_array)
lasso_test_mse_z_shuff_arr = np.array(lasso_test_mse_z_shuff_array)
lasso_test_mse_ct_shuff_ar = np.array(lasso_test_mse_ct_shuff_array)
np.save('abcd_lasso_test_mse_z_arr.npy', lasso_test_mse_z_arr)
np.save('abcd_lasso_test_mse_ct_arr.npy', lasso_test_mse_ct_arr)
np.save('abcd_lasso_test_mse_z_shuff_arr.npy', lasso_test_mse_z_shuff_arr)
np.save('abcd_lasso_test_mse_ct_shuff_arr.npy', lasso_test_mse_ct_shuff_ar)

lasso_perm_arr = np.array(lasso_perm)
lasso_perm_shuff_arr = np.array(lasso_perm_shuff)
np.save('abcd_lasso_perm.npy',lasso_perm_arr)
np.save('abcd_lasso_perm_shuff.npy',lasso_perm_shuff_arr)

ridge_test_mse_z_arr = np.array(ridge_test_mse_z_array)
ridge_test_mse_ct_arr = np.array(ridge_test_mse_ct_array)
ridge_test_mse_z_shuff_arr = np.array(ridge_test_mse_z_shuff_array)
ridge_test_mse_ct_shuff_arr = np.array(ridge_test_mse_ct_shuff_array)
np.save('abcd_ridge_test_mse_z_arr.npy', ridge_test_mse_z_arr)
np.save('abcd_ridge_test_mse_ct_arr.npy', ridge_test_mse_ct_arr)
np.save('abcd_ridge_test_mse_z_shuff_arr.npy', ridge_test_mse_z_shuff_arr)
np.save('abcd_ridge_test_mse_ct_shuff_arr.npy', ridge_test_mse_ct_shuff_arr)

ridge_perm_arr = np.array(ridge_perm)
ridge_perm_shuff_arr = np.array(ridge_perm_shuff)
np.save('abcd_ridge_perm.npy',ridge_perm_arr)
np.save('abcd_ridge_perm_shuff.npy',ridge_perm_shuff)

enet_test_mse_z_arr = np.array(enet_test_mse_z_array)
enet_test_mse_ct_arr = np.array(enet_test_mse_ct_array)
enet_test_mse_z_shuff_arr = np.array(enet_test_mse_z_shuff_array)
enet_test_mse_ct_shuff_arr = np.array(enet_test_mse_ct_shuff_array)
np.save('abcd_enet_test_mse_z_arr.npy', enet_test_mse_z_arr)
np.save('abcd_enet_test_mse_ct_arr.npy', enet_test_mse_ct_arr)
np.save('abcd_enet_test_mse_z_shuff_arr.npy', enet_test_mse_z_shuff_arr)
np.save('abcd_enet_test_mse_ct_shuff_arr.npy', enet_test_mse_ct_shuff_arr)

enet_perm_arr = np.array(enet_perm)
enet_perm_shuff_arr = np.array(enet_perm_shuff)
np.save('abcd_enet_perm.npy',enet_perm_arr)
np.save('abcd_enet_perm_shuff.npy',enet_perm_shuff_arr)