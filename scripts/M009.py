"""
Created by: Rob Mulla
Sep 7

IEEE Fraud Detection Model

- Add back ids
-- FE001
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import matplotlib.pylab as plt
from sklearn.model_selection import KFold
from datetime import datetime
import time
import logging
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from timeit import default_timer as timer
start = timer()

##################
# PARAMETERS
###################
run_id = "{:%m%d_%H%M}".format(datetime.now())
KERNEL_RUN = False
MODEL_NUMBER = os.path.basename(__file__).split('.')[0]

if KERNEL_RUN:
    INPUT_DIR = '../input/champs-scalar-coupling/'
    FE_DIR = '../input/molecule-fe024/'
    FOLDS_DIR = '../input/champs-3fold-ids/'


TARGET = "isFraud"
N_ESTIMATORS = 100000
N_META_ESTIMATORS = 500000
LEARNING_RATE = 0.1
VERBOSE = 1000
EARLY_STOPPING_ROUNDS = 500
RANDOM_STATE = 529
N_THREADS = 48
DEPTH = 7
N_FOLDS = 5

MODEL_TYPE = "catboost"

#####################
## SETUP LOGGER
#####################
def get_logger():
    """
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    os.environ["TZ"] = "US/Eastern"
    time.tzset()
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    fhandler = logging.FileHandler(f'../logs/{MODEL_NUMBER}_{run_id}.log')
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)
#     logger.addHandler(handler)
    logger.addHandler(fhandler)
    return logger

logger = get_logger()

logger.info(f'Running for Model Number {MODEL_NUMBER}')

##################
# PARAMETERS
###################

if MODEL_TYPE == 'xgboost':
    EVAL_METRIC = "AUC"
elif MODEL_TYPE == 'lgbm':
    EVAL_METRIC = 'AUC'
elif MODEL_TYPE == 'catboost':
    EVAL_METRIC = "AUC"

##################
# TRACKING FUNCTION
###################

def update_tracking(run_id,
                    field,
                    value, csv_file="../tracking/tracking.csv", integer=False, digits=None, drop_incomplete_rows=False):
    """
    Function to update the tracking CSV with information about the model
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    if drop_incomplete_rows:
        df = df.loc[~df['AUC'].isna()]
    df.loc[run_id, field] = value  # Model number is index
    df.to_csv(csv_file)

update_tracking(run_id, "model_number", MODEL_NUMBER, drop_incomplete_rows=True)
update_tracking(run_id, "n_estimators", N_ESTIMATORS)
update_tracking(run_id, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)
update_tracking(run_id, "random_state", RANDOM_STATE)
update_tracking(run_id, "n_threads", N_THREADS)
update_tracking(run_id, "learning_rate", LEARNING_RATE)
update_tracking(run_id, "n_fold", N_FOLDS)
update_tracking(run_id, "model_type", MODEL_TYPE)
update_tracking(run_id, "eval_metric", EVAL_METRIC)

#####################
# PREPARE MODEL DATA
#####################
folds = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE)

logger.info('Loading Data...')
train_df = pd.read_parquet('../data/train_FE001.parquet')
test_df = pd.read_parquet('../data/test_FE001.parquet')
logger.info('Done loading Data...')

###########
# FEATURES
###########

FEATURES = ['TransactionAmt', 'ProductCD',
            'card1', 'card2', 'card3',
            'card4', 'card5', 'card6',
            'id_12', 'id_13', 'id_14',
            'id_15', 'id_16', 'id_17',
            'id_18', 'id_19', 'id_20',
            'id_21',
            'id_22',
            'id_23',
            'id_24',
            'id_25',
            'id_26',
            'id_27',
            'id_28',
            'id_29',
            'id_30', 'id_31',
            'id_32',
            'id_33',
            'id_34',
            'id_35',
            'id_36', 'id_37', 'id_38',
            'DeviceType', 'DeviceInfo',
            'M4','P_emaildomain',
            'R_emaildomain',
            'addr1', 'addr2',
            'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
            'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
            'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
            'dist1', 'dist2',
             'DT',
             'DT_M',
             'DT_W',
             'DT_D',
             'DT_hour',
             'DT_day_week',
             'DT_day',
             'D9_fixed',
             'M_sum',
             'M_na',
             'ProductCD_target_mean',
             'M4_target_mean',
             'uid',
             'uid2',
             'uid3',
             'TransactionAmt_check',
             'card1_TransactionAmt_mean',
             'card1_TransactionAmt_std',
             'card2_TransactionAmt_mean',
             'card2_TransactionAmt_std',
             'card3_TransactionAmt_mean',
             'card3_TransactionAmt_std',
             'card5_TransactionAmt_mean',
             'card5_TransactionAmt_std',
             'uid_TransactionAmt_mean',
             'uid_TransactionAmt_std',
             'uid2_TransactionAmt_mean',
             'uid2_TransactionAmt_std',
             'uid3_TransactionAmt_mean',
             'uid3_TransactionAmt_std',
             'TransactionAmt_log1p',
             'email_check',
             'P_emaildomain_prefix',
             'R_emaildomain_prefix',
             'DeviceInfo_device',
             'DeviceInfo_version',
             'id_30_device',
             'id_30_version',
             'id_31_device',
             'card1_fq_enc',
             'card2_fq_enc',
             'card3_fq_enc',
             'card5_fq_enc',
             'C1_fq_enc',
             'C2_fq_enc',
             'C3_fq_enc',
             'C4_fq_enc',
             'C5_fq_enc',
             'C6_fq_enc',
             'C7_fq_enc',
             'C8_fq_enc',
             'C9_fq_enc',
             'C10_fq_enc',
             'C11_fq_enc',
             'C12_fq_enc',
             'C13_fq_enc',
             'C14_fq_enc',
             'D1_fq_enc',
             'D2_fq_enc',
             'D3_fq_enc',
             'D4_fq_enc',
             'D5_fq_enc',
             'D6_fq_enc',
             'D7_fq_enc',
             'D8_fq_enc',
             'addr1_fq_enc',
             'addr2_fq_enc',
             'dist1_fq_enc',
             'dist2_fq_enc',
             'P_emaildomain_fq_enc',
             'R_emaildomain_fq_enc',
             'DeviceInfo_fq_enc',
             'DeviceInfo_device_fq_enc',
             'DeviceInfo_version_fq_enc',
             'id_30_fq_enc',
             'id_30_device_fq_enc',
             'id_30_version_fq_enc',
             'id_31_device_fq_enc',
             'id_33_fq_enc',
             'uid_fq_enc',
             'uid2_fq_enc',
             'uid3_fq_enc',
             'DT_M_total',
             'DT_W_total',
             'DT_D_total',
             'uid_DT_M',
             'uid_DT_W',
             'uid_DT_D']

CAT_FEATURES = ['ProductCD', 'card4', 'card6',
            'id_12', 'id_13', 'id_14',
            'id_15', 'id_16', 'id_17',
            'id_18', 'id_19', 'id_20',
            'id_21',
            'id_22',
            'id_23',
            'id_24',
            'id_25',
            'id_26',
            'id_27',
            'id_28',
            'id_29',
            'id_30', 'id_31',
            'id_32',
            'id_33',
            'id_34',
            'id_35',
            'id_36', 'id_37', 'id_38',
            'DeviceType', 'DeviceInfo',
            'M4','P_emaildomain',
            'R_emaildomain', 'addr1', 'addr2',
            'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]

X = X.fillna(-9999)
X_test = X_test.fillna(-9999)

logger.info('Running with features...')
logger.info(FEATURES)
logger.info(f'Target is {TARGET}')


update_tracking(run_id, "n_features", len(FEATURES), integer=True)
################################
# Dataframes for storing results
#################################

feature_importance = pd.DataFrame()
oof = np.zeros(len(X))
pred = np.zeros(len(X_test))
oof_df = train_df[['isFraud']].copy()
oof_df['oof'] = np.nan
oof_df['fold'] = np.nan
scores = []
best_iterations = []

for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    if MODEL_TYPE == "catboost":
        train_dataset = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
        valid_dataset = Pool(data=X_valid, label=y_valid, cat_features=CAT_FEATURES)
        test_dataset = Pool(data=X_test, cat_features=CAT_FEATURES)

        model = CatBoostClassifier(
                iterations=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                depth=DEPTH,
                eval_metric=EVAL_METRIC,
                verbose=VERBOSE,
                random_state=RANDOM_STATE,
                thread_count=N_THREADS,
                task_type="GPU")

        model.fit(
                train_dataset,
                eval_set=valid_dataset,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
        y_pred_valid = model.predict_proba(valid_dataset)[:,1]
        y_pred = model.predict_proba(test_dataset)[:,1]

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = model.feature_names_
        fold_importance["importance"] = model.get_feature_importance()
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance],
                                       axis=0)
        best_iteration = model.best_iteration_
    best_iterations.append(best_iteration)

    fold_score = roc_auc_score(y_valid, y_pred_valid)
    scores.append(fold_score)

    update_tracking(run_id, "AUC_f{}".format(fold_n + 1),
                    fold_score,
                    integer=False,)
    logger.info('Fold {} of {} CV mean AUC score: {:.4f}. Best iteration {}'.format(fold_n + 1,
                                                                  N_FOLDS,
                                                                  fold_score,
                                                                  best_iteration))
    oof_df.iloc[valid_idx, oof_df.columns.get_loc('oof')] = y_pred_valid.reshape(-1)
    oof_df.iloc[valid_idx, oof_df.columns.get_loc('fold')] = fold_n + 1
    pred += y_pred

update_tracking(run_id, 'avg_best_iteration',
                np.mean(best_iterations),
                integer=True)

###############
# Store Results
###############
pred /= N_FOLDS
score = np.mean(scores)
sub = pd.read_csv('../input/sample_submission.csv')
sub['isFraud'] = pred
sub.to_csv(f'../sub/sub_{MODEL_NUMBER}_{run_id}_{score:.4f}.csv', index=False)
oof_df.to_csv(f'../oof/oof_{MODEL_NUMBER}_{run_id}_{score:.4f}.csv')
logger.info('CV mean AUC score: {:.4f}, std: {:.4f}.'.format(np.mean(scores),
                                                             np.std(scores)))
total_score = roc_auc_score(oof_df['isFraud'], oof_df['oof'])
feature_importance.to_csv(f'../fi/fi_{MODEL_NUMBER}_{run_id}_{score:.4f}.csv')

update_tracking(run_id, "AUC",
                total_score,
                integer=False,)
logger.info('OOF AUC Score: {:.4f}'.format(total_score))
end = timer()
update_tracking(run_id, "training_time", (end - start), integer=True)
logger.info('Done!')
