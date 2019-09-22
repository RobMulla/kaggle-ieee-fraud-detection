"""
Created by: Rob Mulla
Sep 21

IEEE Fraud Detection Model

- FE009
- Adding raddar user level features
- Add first, second, third digit of addr1 and addr2 features
- Drop only DOY features with low importance
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
LEARNING_RATE = 0.005
VERBOSE = 100
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE = 529
N_THREADS = 58
DEPTH = 14
N_FOLDS = 5
SHUFFLE = False
FE_SET = 'FE009' # Feature Engineering Version

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
update_tracking(run_id, "fe", FE_SET)

#####################
# PREPARE MODEL DATA
#####################
folds = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=SHUFFLE)

logger.info('Loading Data...')
train_df = pd.read_parquet(f'../data/train_{FE_SET}.parquet')
test_df = pd.read_parquet(f'../data/test_{FE_SET}.parquet')
logger.info('Done loading Data...')

###########
# FEATURES
###########

REMOVE_FEATURES = ['TransactionID', 'TransactionDT', 'isFraud', 'DT', 'DT_M', 'DT_W',
       'DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month', 'DT_M_total',
       'DT_W_total', 'DT_D_total', 'uid', 'uid2', 'uid3', 'uid4', 'uid5',
       'bank_type', 'year_day', 'V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120','Day_of_Year_Year']

LOW_IMPORTANCE_FEATS = [
    'dist2_div_Mean_D11_DOY',  'dist1_div_Mean_D8_DOY',
    'dist1_div_Mean_D9_DOY_productCD',  'dist1_div_Mean_D8_DOY_productCD',
    'dist2_div_Mean_D11_DOY_productCD', 'dist1_div_Mean_D7_DOY',
    'dist1_div_Mean_D6_DOY_productCD',  'dist1_div_Mean_D6_DOY',
    'dist1_div_Mean_D7_DOY_productCD', 'dist1_div_Mean_D9_DOY',
    'dist1_div_Mean_D14_DOY_productCD',  'dist1_div_Mean_D14_DOY',
    'dist1_div_Mean_D13_DOY_productCD',  'dist1_div_Mean_D13_DOY',
    'dist1_div_Mean_D12_DOY_productCD',  'dist1_div_Mean_D12_DOY',
    'addr2_div_Mean_D7_DOY_productCD',  'addr2_div_Mean_D8_DOY_productCD',
    'addr2_div_Mean_D12_DOY_productCD',  'addr2_div_Mean_D13_DOY_productCD',
    'addr2_div_Mean_D5_DOY_productCD',  'addr1_div_Mean_D12_DOY_productCD',
    'addr2_div_Mean_D8_DOY',  'addr2_div_Mean_D2_DOY_productCD',
    'addr2_div_Mean_D6_DOY_productCD',  'addr2_div_Mean_D12_DOY',
    'addr2_div_Mean_D7_DOY',  'addr1_div_Mean_D7_DOY',
    'addr1_div_Mean_D7_DOY_productCD',  'addr2_div_Mean_D3_DOY_productCD',
    'addr2_div_Mean_D14_DOY_productCD',  'addr1_div_Mean_D12_DOY',
    'addr2_div_Mean_D13_DOY',  'addr1_div_Mean_D13_DOY',
    'dist2_div_Mean_D12_DOY_productCD',  'dist2_div_Mean_D8_DOY_productCD',
    'dist2_div_Mean_D2_DOY_productCD',  'addr1_div_Mean_D13_DOY_productCD',
    'addr2_div_Mean_D9_DOY_productCD',  'dist2_div_Mean_D7_DOY_productCD',
    'addr1_div_Mean_D6_DOY',
    'addr1_div_Mean_D8_DOY_productCD',  'addr1_div_Mean_D6_DOY_productCD',
    'dist2_div_Mean_D5_DOY_productCD',
    'dist2_div_Mean_D3_DOY_productCD',  'dist1_div_Mean_D5_DOY_productCD',
    'dist2_div_Mean_D6_DOY_productCD', 
    'addr1_div_Mean_D8_DOY',  'dist1_div_Mean_D2_DOY_productCD',
    'addr2_div_Mean_D11_DOY_productCD',  'addr2_div_Mean_D14_DOY',
    'dist2_div_Mean_D2_DOY',  'dist2_div_Mean_D8_DOY',
    'card5_div_Mean_D12_DOY_productCD',
    'dist2_div_Mean_D4_DOY_productCD',  'dist1_div_Mean_D3_DOY_productCD',
    'card6_div_Mean_D12_DOY_productCD',  'dist2_div_Mean_D3_DOY',
    'card1_div_Mean_D12_DOY_productCD',
    'card1_div_Mean_D7_DOY_productCD',  'card3_div_Mean_D2_DOY_productCD',
    'card3_div_Mean_D8_DOY_productCD','dist1_div_Mean_D1_DOY_productCD',
    'card4_div_Mean_D12_DOY_productCD',  'card4_div_Mean_D7_DOY_productCD',
    'card3_div_Mean_D12_DOY_productCD', 'dist2_div_Mean_D5_DOY','dist1_div_Mean_D4_DOY_productCD',
    'addr2_div_Mean_D2_DOY', 'card6_div_Mean_D7_DOY_productCD',
    'addr1_div_Mean_D14_DOY_productCD', 'card2_div_Mean_D12_DOY_productCD',
    'card2_div_Mean_D7_DOY_productCD',  'card5_div_Mean_D7_DOY_productCD',
    'dist2_div_Mean_D12_DOY',
    'dist1_div_Mean_D11_DOY_productCD',
    'dist1_div_Mean_D10_DOY_productCD',
    'card1_div_Mean_D7_DOY',
    'card3_div_Mean_D7_DOY_productCD',
    'card6_div_Mean_D8_DOY',
    'addr1_div_Mean_D9_DOY',  'card5_div_Mean_D7_DOY',
    'dist2_div_Mean_D4_DOY', 'card2_div_Mean_D7_DOY',
    'dist2_div_Mean_D10_DOY_productCD', 'addr2_div_Mean_D5_DOY',
    'TranAmt_div_Mean_D8_DOY_productCD',
    'dist2_div_Mean_D6_DOY',
    'card4_div_Mean_D7_DOY',
    'addr1_div_Mean_D9_DOY_productCD',
    'addr1_div_Mean_D11_DOY_productCD',
    'dist2_div_Mean_D9_DOY_productCD',
    'TranAmt_div_Mean_D7_DOY_productCD',
    'addr2_div_Mean_D3_DOY',
    'card1_div_Mean_D11_DOY_productCD',
    'card6_div_Mean_D7_DOY',
    'dist2_div_Mean_D11_DOY', 'dist1_div_Mean_D8_DOY', 'dist1_div_Mean_D9_DOY_productCD',
     'dist1_div_Mean_D8_DOY_productCD',
     'dist2_div_Mean_D11_DOY_productCD',
     'dist1_div_Mean_D7_DOY',
     'dist1_div_Mean_D6_DOY_productCD',
     'dist1_div_Mean_D6_DOY',
     'dist1_div_Mean_D7_DOY_productCD',
     'dist1_div_Mean_D9_DOY',
     'dist1_div_Mean_D14_DOY_productCD',
     'dist1_div_Mean_D14_DOY',
     'dist1_div_Mean_D13_DOY_productCD',
     'dist1_div_Mean_D13_DOY',
     'dist1_div_Mean_D12_DOY_productCD',
     'dist1_div_Mean_D12_DOY',
     'addr2_div_Mean_D7_DOY_productCD',
     'addr2_div_Mean_D8_DOY_productCD',
     'addr2_div_Mean_D12_DOY_productCD',
     'addr2_div_Mean_D13_DOY_productCD',
     'addr2_div_Mean_D5_DOY_productCD',
     'addr1_div_Mean_D12_DOY_productCD',
     'addr2_div_Mean_D8_DOY',
     'addr2_div_Mean_D2_DOY_productCD',
     'addr2_div_Mean_D6_DOY_productCD',
     'addr2_div_Mean_D12_DOY',
     'addr2_div_Mean_D7_DOY',
     'addr1_div_Mean_D7_DOY',
     'addr1_div_Mean_D7_DOY_productCD',
     'addr2_div_Mean_D3_DOY_productCD',
     'addr2_div_Mean_D14_DOY_productCD',
     'addr1_div_Mean_D12_DOY',
     'addr2_div_Mean_D13_DOY',
     'addr1_div_Mean_D13_DOY',
     'dist2_div_Mean_D12_DOY_productCD',
     'dist2_div_Mean_D8_DOY_productCD',
     'dist2_div_Mean_D2_DOY_productCD',
     'addr1_div_Mean_D13_DOY_productCD',
     'addr2_div_Mean_D9_DOY_productCD',
     'dist2_div_Mean_D7_DOY_productCD',
     'addr1_div_Mean_D6_DOY',
     'addr1_div_Mean_D8_DOY_productCD',
     'addr1_div_Mean_D6_DOY_productCD',
     'dist2_div_Mean_D5_DOY_productCD',
     'dist2_div_Mean_D3_DOY_productCD',
     'dist1_div_Mean_D5_DOY_productCD',
     'dist2_div_Mean_D6_DOY_productCD',
     'addr1_div_Mean_D8_DOY',
     'addr2_div_Mean_D6_DOY',
     'dist1_div_Mean_D2_DOY_productCD',
     'addr2_div_Mean_D11_DOY_productCD',
     'addr2_div_Mean_D14_DOY',
     'dist2_div_Mean_D2_DOY',
     'dist2_div_Mean_D8_DOY',
     'card5_div_Mean_D12_DOY_productCD',
     'dist2_div_Mean_D4_DOY_productCD',
     'dist1_div_Mean_D3_DOY_productCD',
     'card6_div_Mean_D12_DOY_productCD',
     'dist2_div_Mean_D3_DOY',
'TranAmt_div_Mean_D11_DOY_productCD',
 'card5_div_Mean_D8_DOY',
 'card5_div_Mean_D8_DOY_productCD',
 'card3_div_Mean_D11_DOY_productCD',
 'card5_div_Mean_D11_DOY_productCD',
 'card4_div_Mean_D11_DOY_productCD',
 'card6_div_Mean_D11_DOY_productCD',
 'dist2_div_Mean_D14_DOY_productCD',
 'card2_div_Mean_D8_DOY_productCD',
 'card1_div_Mean_D8_DOY',
 'TranAmt_div_Mean_D8_DOY',
 'card2_div_Mean_D11_DOY_productCD',
 'card2_div_Mean_D8_DOY',
 'card4_div_Mean_D8_DOY',
 'card3_div_Mean_D6_DOY_productCD',
 'dist1_div_Mean_D2_DOY',
 'card6_div_Mean_D8_DOY_productCD',
 'card1_div_Mean_D6_DOY_productCD',
 'card1_div_Mean_D8_DOY_productCD',
 'dist2_div_Mean_D13_DOY_productCD',
 'dist1_div_Mean_D5_DOY',
 'dist2_div_Mean_D1_DOY_productCD',
 'card3_div_Mean_D7_DOY',
 'card3_div_Mean_D8_DOY',
 'addr1_div_Mean_D5_DOY_productCD',
 'card4_div_Mean_D9_DOY_productCD',
 'card1_div_Mean_D9_DOY',
 'card4_div_Mean_D6_DOY_productCD',
 'addr1_div_Mean_D2_DOY_productCD',
 'dist2_div_Mean_D1_DOY',
 'card2_div_Mean_D9_DOY_productCD',
 'TranAmt_div_Mean_D7_DOY',
 'card3_div_Mean_D3_DOY_productCD',
 'card1_div_Mean_D13_DOY_productCD',
 'TranAmt_div_Mean_D6_DOY_productCD',
 'card6_div_Mean_D6_DOY_productCD',
 'card6_div_Mean_D2_DOY_productCD',
 'dist1_div_Mean_D3_DOY',
 'card5_div_Mean_D6_DOY_productCD',
 'dist2_div_Mean_D9_DOY',
 'dist2_div_Mean_D14_DOY',
 'card2_div_Mean_D9_DOY',
 'addr2_div_Mean_D11_DOY',
 'card4_div_Mean_D5_DOY_productCD',
 'dist2_div_Mean_D10_DOY',
 'card1_div_Mean_D6_DOY',
 'card1_div_Mean_D9_DOY_productCD',
 'addr1_div_Mean_D5_DOY',
 'card4_div_Mean_D9_DOY',
 'addr1_div_Mean_D3_DOY_productCD',
 'dist2_div_Mean_D13_DOY',
 'card3_div_Mean_D9_DOY_productCD',
 'card5_div_Mean_D9_DOY_productCD',
 'addr1_div_Mean_D2_DOY',
 'card3_div_Mean_D5_DOY_productCD',
 'TranAmt_div_Mean_D2_DOY_productCD',
 'card5_div_Mean_D13_DOY_productCD',
 'card1_div_Mean_D12_DOY',
 'card2_div_Mean_D6_DOY_productCD',
 'card5_div_Mean_D9_DOY',
 'card5_div_Mean_D12_DOY',
 'card5_div_Mean_D5_DOY_productCD',
 'card5_div_Mean_D6_DOY',
 'card6_div_Mean_D9_DOY_productCD',
 'TranAmt_div_Mean_D3_DOY_productCD',
 'addr1_div_Mean_D3_DOY',
 'card1_div_Mean_D14_DOY_productCD',
 'TranAmt_div_Mean_D2_DOY',
 'card6_div_Mean_D5_DOY_productCD',
 'addr2_div_Mean_D4_DOY_productCD',
 'TranAmt_div_Mean_D5_DOY_productCD',
 'card2_div_Mean_D13_DOY_productCD',
 'card1_div_Mean_D5_DOY_productCD',
 'card4_div_Mean_D13_DOY_productCD',
 'card1_div_Mean_D13_DOY',
 'card2_div_Mean_D14_DOY_productCD',
 'card5_div_Mean_D2_DOY_productCD',
 'TranAmt_div_Mean_D5_DOY',
 'card5_div_Mean_D13_DOY',
 'card2_div_Mean_D2_DOY_productCD',
 'card6_div_Mean_D12_DOY',
 'TranAmt_div_Mean_D3_DOY',
 'card5_div_Mean_D2_DOY',
 'card6_div_Mean_D9_DOY',
 'card4_div_Mean_D6_DOY',
 'card1_div_Mean_D2_DOY_productCD',
 'card2_div_Mean_D5_DOY_productCD',
 'card5_div_Mean_D5_DOY',
 'card4_div_Mean_D2_DOY_productCD',
 'card3_div_Mean_D3_DOY',
 'card6_div_Mean_D6_DOY',
 'card4_div_Mean_D12_DOY',
 'card4_div_Mean_D5_DOY',
 'card1_div_Mean_D5_DOY',
 'card4_div_Mean_D14_DOY_productCD',
 'card4_div_Mean_D2_DOY',
 'card6_div_Mean_D5_DOY',
 'card6_div_Mean_D3_DOY_productCD',
 'card3_div_Mean_D5_DOY',
 'card1_div_Mean_D14_DOY',
 'card2_div_Mean_D3_DOY_productCD',
 'card2_div_Mean_D2_DOY',
 'card2_div_Mean_D5_DOY',
 'card3_div_Mean_D2_DOY',
 'TranAmt_div_Mean_D13_DOY_productCD',
 'card2_div_Mean_D14_DOY',
 'card6_div_Mean_D13_DOY_productCD',
 'dist1_div_Mean_D1_DOY',
 'card2_div_Mean_D6_DOY',
 'card5_div_Mean_D3_DOY_productCD',
 'dist1_div_Mean_D4_DOY',
 'card4_div_Mean_D13_DOY',
 'dist1_div_Mean_D11_DOY',
 'card6_div_Mean_D14_DOY_productCD',
 'card6_div_Mean_D2_DOY',
 'card6_div_Mean_D14_DOY',
 'card2_div_Mean_D12_DOY',
 'card2_div_Mean_D13_DOY',
 'TranAmt_div_Mean_D9_DOY_productCD',
 'card3_div_Mean_D13_DOY_productCD',
 'card1_div_Mean_D3_DOY_productCD',
 'card4_div_Mean_D3_DOY_productCD',
 'card6_div_Mean_D13_DOY',
 'card4_div_Mean_D14_DOY',
 'dist1_div_Mean_D10_DOY',
 'TranAmt_div_Mean_D13_DOY',
 'TranAmt_div_Mean_D14_DOY_productCD',
 'TranAmt_div_Mean_D12_DOY',
 'TranAmt_div_Mean_D6_DOY',
 'TranAmt_div_Mean_D9_DOY',
 'card5_div_Mean_D3_DOY',
 'card4_div_Mean_D3_DOY',
 'card3_div_Mean_D12_DOY',
 'card5_div_Mean_D14_DOY',
 'card3_div_Mean_D13_DOY',
 'card5_div_Mean_D14_DOY_productCD',
 'addr1_div_Mean_D4_DOY_productCD',
 'card1_div_Mean_D2_DOY',
 'card3_div_Mean_D14_DOY_productCD',
 'card2_div_Mean_D3_DOY',
 'card1_div_Mean_D3_DOY',
 'TranAmt_div_Mean_D11_DOY']

REMOVE_FEATURES = REMOVE_FEATURES + LOW_IMPORTANCE_FEATS

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
            'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'ProductCD_W_95cents','ProductCD_W_00cents','ProductCD_W_50cents',
            'ProductCD_W_50_95_0_cents','ProductCD_W_NOT_50_95_0_cents']

CAT_FEATURES = [c for c in CAT_FEATURES if c not in REMOVE_FEATURES]

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
                'learning_rate':LEARNING_RATE,
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
