"""
Created by: Rob Mulla
Sep 10

IEEE Fraud Detection Model

- Features from public kernel
- Shuffle = False
- lgbm
- Remove Low feature importance

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
import lightgbm as lgb

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
VERBOSE = 100
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE = 529
N_THREADS = 48
DEPTH = 14
N_FOLDS = 5
SHUFFLE = False

MODEL_TYPE = "lightgbm"

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
elif MODEL_TYPE == 'lightgbm':
    EVAL_METRIC = 'auc'
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
update_tracking(run_id, "depth", DEPTH)
update_tracking(run_id, "shuffle", SHUFFLE)


#####################
# PREPARE MODEL DATA
#####################
folds = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=SHUFFLE)

logger.info('Loading Data...')
train_df = pd.read_parquet('../data/train_FE003.parquet')
test_df = pd.read_parquet('../data/test_FE003.parquet')
logger.info('Done loading Data...')

###########
# FEATURES
###########

REMOVE_FEATURES = ['TransactionID', 'TransactionDT', 'isFraud', 'DT', 'DT_M', 'DT_W',
       'DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month', 'DT_M_total',
       'DT_W_total', 'DT_D_total', 'uid', 'uid2', 'uid3', 'uid4', 'uid5',
       'bank_type', 'year_day']

# From Previous Run Feature Importance
LOW_IMPORTANCE_FEATURES = \
     ['D8_not_same_day', 'V120', 'V119', 'V305', 'D9_fixed', 'V28', 'M_sum',
        'addr2', 'id_22', 'id_24', 'V68', 'D9_not_na', 'V27', 'V1', 'V89', 'V117',
        'V118', 'V88', 'V269', 'V41', 'V240', 'V241', 'V107', 'V32', 'id_26', 'V113',
        'V14', 'V325', 'V122', 'V21', 'V31', 'id_27', 'V16', 'V50', 'id_35', 'id_29',
        'V328', 'V334', 'V106', 'V185', 'V112', 'V157', 'V196', 'V104', 'V284', 'V98',
        'V18', 'V236', 'V22', 'id_17', 'V15', 'V121', 'V71', 'V297', 'V337', 'V84',
        'V141', 'V226', 'V173', 'V193', 'V299', 'V114', 'V111', 'V174', 'V175',
        'V100', 'V110', 'V92', 'V116', 'V252', 'V158', 'V179', 'V331', 'M1', 'V101',
        'V330', 'V195', 'V103', 'V247', 'V191', 'V327', 'V142', 'V148', 'V153',
        'id_30_device', 'V322', 'V144', 'V17', 'D7', 'V138', 'V186', 'V237', 'V231',
        'V183', 'V151', 'V190', 'V302', 'id_04', 'V199', 'V339', 'id_18', 'V97',
        'V181', 'id_12', 'D7_DT_M_min_max', 'V225', 'V250', 'addr2_fq_enc', 'V168',
        'V329', 'V95', 'V326', 'V9', 'V177', 'V249', 'is_holiday', 'V255', 'V184',
        'V254', 'V239', 'V155', 'V219', 'V167', 'TransactionAmt_check', 'id_10',
        'V59', 'V146', 'V238', 'V194', 'V228', 'V235', 'V182', 'V336', 'V93', 'V57',
        'V335', 'V192', 'V288', 'V163', 'V80', 'V125', 'V338', 'V140',
        'D7_DT_W_min_max', 'V176', 'V65', 'V108', 'card3_TransactionAmt_std', 'V230',
        'id_28', 'V123', 'V46', 'V73', 'D12_DT_M_min_max', 'V211', 'V286', 'V287',
        'V42', 'V333', 'V72', 'D6_DT_M_min_max', 'V159', 'V218', 'D7_DT_D_min_max',
        'V295', 'V161', 'V2', 'V63', 'V172', 'V8', 'D7_fq_enc', 'V198', 'id_14',
        'V242', 'V115', 'card3_TransactionAmt_mean', 'V332', 'V272', 'V233', 'V214',
        'V227', 'D13_DT_M_min_max', 'V105', 'V202', 'V3', 'V64', 'V34', 'card4',
        'D14_DT_M_min_max', 'V134', 'V145', 'V213', 'uid5_D12_std', 'D6_DT_W_min_max',
        'id_32', 'V298', 'V51', 'V246', 'V171', 'V289', 'V204', 'V94', 'V324', 'V224',
        'V39', 'id_30_device_fq_enc', 'uid5_D12_mean', 'V33', 'V292',
        'D7_DT_M_std_score', 'V137', 'card3', 'V154', 'id_34', 'V109', 'V180', 'V248',
        'V273', 'V244', 'V263', 'V275', 'V278', 'V301', 'V25', 'V132', 'V319', 'V178',
        'V102', 'id_08', 'C7_fq_enc', 'V232', 'V197', 'bank_type_D11_std', 'V206',
        'V188', 'V170', 'V223', 'V11', 'V303', 'V135', 'V304', 'bank_type_D8_std',
        'V260', 'V300', 'V26', 'C3_fq_enc', 'V253', 'id_25', 'V10', 'V43', 'V217',
        'V60', 'V276', 'V222', 'D13', 'card3_fq_enc', 'V6', 'V7', 'V74', 'V4', 'V321',
        'V210', 'id_16', 'V69', 'V85', 'bank_type_D11_mean', 'D12_DT_W_min_max',
        'V216', 'V162', 'V290', 'V265', 'V212', 'V150', 'V52', 'D5_DT_M_min_max',
        'V40', 'V268', 'D6', 'V147', 'V30', 'V293', 'V126', 'D12', 'V229',
        'D13_DT_W_min_max', 'V280', 'id_23', 'V166', 'C7', 'id_36',
        'bank_type_D15_mean', 'bank_type_D15_std', 'V251', 'V58', 'V270',
        'bank_type_D4_std', 'V323', 'V160', 'V262', 'D12_DT_D_min_max', 'id_07',
        'V318', 'id_37', 'V5', 'V29', 'C4_fq_enc', 'V200', 'id_21', 'V79', 'V205',
        'V215', 'V234', 'V35', 'V245', 'V201', 'V139', 'V96', 'D6_DT_D_min_max',
        'V164', 'C3', 'V209', 'V81', 'D5_DT_W_min_max', 'D5', 'V207', 'V49', 'V23',
        'V129', 'V203', 'D7_DT_W_std_score', 'V36', 'V99']

REMOVE_FEATURES = REMOVE_FEATURES + LOW_IMPORTANCE_FEATURES

FEATURES = [c for c in test_df.columns if c not in REMOVE_FEATURES]

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
            'id_32',
            'id_34',
            'id_35',
            'id_36', 'id_37', 'id_38',
            'DeviceType', 'DeviceInfo',
            'M4','P_emaildomain',
            'R_emaildomain', 'addr1', 'addr2',
            'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

CAT_FEATURES = [c for c in CAT_FEATURES if c in FEATURES]

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]

X = X.fillna(-9999)
X_test = X_test.fillna(-9999)

logger.info('Running with features...')
logger.info(FEATURES)
logger.info(f'Target is {TARGET}')


update_tracking(run_id, "n_features", len(FEATURES), integer=True)


############################
#### TRAIN MODELS FUNCTIONS
############################

def train_catboost(X_train, y_train, X_valid, y_valid, X_test, CAT_FEATURES, fold_n, feature_importance):
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
    return y_pred, y_pred_valid, feature_importance, best_iteration


lgb_params = {
                'objective':'binary',
                'boosting_type':'gbdt',
                'metric': EVAL_METRIC,
                'n_jobs':N_THREADS,
                'learning_rate':0.01,
                'num_leaves': 2**8,
                'max_depth':DEPTH,
                'tree_learner':'serial',
                'colsample_bytree': 0.85,
                'subsample_freq':1,
                'subsample':0.85,
                'n_estimators':N_ESTIMATORS,
                'max_bin':255,
                'verbose':-1,
                'seed': RANDOM_STATE,
                #'early_stopping_rounds':EARLY_STOPPING_ROUNDS,
                'reg_alpha':0.3,
                'reg_lamdba':0.243,
                #'categorical_feature': CAT_FEATURES
            }

def train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, CAT_FEATURES, fold_n, feature_importance):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    X_train[CAT_FEATURES] = X_train[CAT_FEATURES].astype('category')
    X_valid[CAT_FEATURES] = X_valid[CAT_FEATURES].astype('category')
    X_test[CAT_FEATURES] = X_test[CAT_FEATURES].astype('category')

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(X_train, y_train,
            eval_set = [(X_train, y_train),
                        (X_valid, y_valid)],
            verbose = VERBOSE,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    y_pred_valid = model.predict_proba(X_valid)[:,1]
    y_pred = model.predict_proba(X_test)[:,1]

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X_train.columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance],
                                   axis=0)
    best_iteration = model.best_iteration_
    return y_pred, y_pred_valid, feature_importance, best_iteration

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
        y_pred, y_pred_valid, feature_importance, best_iteration = train_catboost(X_train, y_train, X_valid, y_valid, X_test, CAT_FEATURES, fold_n, feature_importance)
    if MODEL_TYPE == 'lightgbm':
        y_pred, y_pred_valid, feature_importance, best_iteration = train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, CAT_FEATURES, fold_n, feature_importance)
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
