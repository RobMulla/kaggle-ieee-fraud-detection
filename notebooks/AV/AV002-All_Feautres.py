import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

tr = pd.read_parquet('../../data/train_FE013.parquet')

COLS_TO_DROP = ['D3_intercept_bin0',
         'D14_intercept_bin0',
         'D13_intercept_bin0',
         'D11_intercept_bin2',
         'D4_intercept_bin3',
         'D2_intercept_bin3',
         'D4_intercept_bin4',
         'D11_intercept_bin0',
         'D8_intercept_bin0',
         'D4_intercept_bin1',
         'D10_intercept_bin3',
         'D4_intercept_bin2',
         'D2_intercept_bin2',
         'D15_intercept_bin4',
         'D10_intercept_bin4',
         'D15_intercept_bin2',
         'D10_intercept_bin2',
         'D1_intercept_bin4',
         'D2_intercept_bin1',
         'D10_intercept_bin1',
         'D15_intercept_bin3',
         'D1_intercept_bin3',
         'D15_intercept_bin1',
         'D2_intercept_bin0',
         'D4_intercept_bin0',
         'D1_intercept_bin1',
         'D10_intercept_bin0',
         'D1_intercept_bin2',
         'D15_intercept_bin0',
         'D1_intercept_bin0']

features = [c for c in tr.columns if c not in ['i_am_train', # This is the actual target for adv validation
                                               'userid', # An ID
                                               'isFraud', # The target for competition
                                                    'TransactionID','TransactionDT'
                                              ]]

del tr
gc.collect()

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

lgb_params = {
    'task': 'train',
    'max_depth': 10,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 20,
    'learning_rate': 0.05,
    'feature_fraction': 0.50,
    'bagging_fraction': 0.81,
    'bagging_freq': 1,
    'lambda_l1': 3,
    'lambda_l2': 3,
    'verbose': -99,
    'boost_from_average': 'true',
    'seed': 99,
    'nthreads': 4
}

for f in tqdm(features):
    output_df = pd.read_csv('av002-output.csv')
    if f in output_df['feature'].tolist():
        print(f'Already run for feature {f}. skipping....')
        continue
    else:
        output_df = output_df.set_index('feature')
        output_df.loc[f,'cv'] = 'Running'
        output_df = output_df.reset_index()
        output_df.to_csv('av002-output.csv', index=False)
    tr = pd.read_parquet('../../data/train_FE013.parquet', columns=[f, 'TransactionID'])
    te = pd.read_parquet('../../data/test_FE013.parquet', columns=[f, 'TransactionID'])
    tr['i_am_train'] = 1
    te['i_am_train'] = 0
    te = te.loc[te['TransactionID']>3764887]

    full_df = pd.concat([tr, te], axis=0, sort=True).reset_index(drop=True)

    oof_preds = np.zeros(full_df.shape[0])
    print('Fitting to feature',f)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(full_df.values, full_df['i_am_train'].values)):
        #print("Fold idx:{}".format(fold_ + 1))

        X_train, y_train = pd.DataFrame(full_df.iloc[trn_idx][f]), full_df['i_am_train'].iloc[trn_idx].values
        X_valid, y_valid = pd.DataFrame(full_df.iloc[val_idx][f]), full_df['i_am_train'].iloc[val_idx].values

        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(lgb_params, trn_data, 800, valid_sets = [trn_data, val_data],
                        verbose_eval=0, early_stopping_rounds = 100)

        oof_preds[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
    output_df = pd.read_csv('av002-output.csv')
    output_df = output_df.set_index('feature')
    output_df.loc[f, 'cv'] = roc_auc_score(full_df['i_am_train'], oof_preds)
    output_df.loc[f, 'best_iter'] = clf.best_iteration
    output_df = output_df.reset_index()
    output_df.to_csv('av002-output.csv', index=False)
