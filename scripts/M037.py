"""
Created by: Rob Mulla
Sep 24

IEEE Fraud Detection Model

- FE013
- Yang's Features
- Raddars Features

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
import gc

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
DEPTH = -1 #14
N_FOLDS = 5
SHUFFLE = False
FE_SET = 'FE013' # Feature Engineering Version

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

FEATURES = ['V1max', 'V2max', 'V3max', 'V4max', 'V5max', 'V6max', 'V7max',
 'V8max', 'V9max', 'V10max', 'V11max', 'V12max', 'V13max', 'V14max', 'V15max',
 'V16max', 'V17max', 'V18max', 'V19max', 'V20max', 'V21max', 'V22max',
 'V23max', 'V24max', 'V25max', 'V26max', 'V27max', 'V28max', 'V29max',
 'V30max', 'V31max', 'V32max', 'V33max', 'V34max', 'V35max', 'V36max',
 'V37max', 'V38max', 'V39max', 'V40max', 'V41max', 'V42max', 'V43max',
 'V44max', 'V45max', 'V46max', 'V47max', 'V48max', 'V49max', 'V50max',
 'V51max', 'V52max', 'V53max', 'V54max', 'V55max', 'V56max', 'V57max',
 'V58max', 'V59max', 'V60max', 'V61max', 'V62max', 'V63max', 'V64max',
 'V65max', 'V66max', 'V67max', 'V68max', 'V69max', 'V70max', 'V71max',
 'V72max', 'V73max', 'V74max', 'V75max', 'V76max', 'V77max', 'V78max',
 'V79max', 'V80max', 'V81max', 'V82max', 'V83max', 'V84max', 'V85max',
 'V86max', 'V87max', 'V88max', 'V89max', 'V90max', 'V91max', 'V92max',
 'V93max', 'V94max', 'V95max', 'V96max', 'V97max', 'V98max', 'V99max',
 'V100max', 'V101max', 'V102max', 'V103max', 'V104max', 'V105max', 'V106max',
 'V107max', 'V108max', 'V109max', 'V110max', 'V111max', 'V112max', 'V113max',
 'V114max', 'V115max', 'V116max', 'V117max', 'V118max', 'V119max', 'V120max',
 'V121max', 'V122max', 'V123max', 'V124max', 'V125max', 'V126max', 'V127max',
 'V128max', 'V129max', 'V130max', 'V131max', 'V132max', 'V133max', 'V134max',
 'V135max', 'V136max', 'V137max', 'V138max', 'V139max', 'V140max', 'V141max',
 'V142max', 'V143max', 'V144max', 'V145max', 'V146max', 'V147max', 'V148max',
 'V149max', 'V150max', 'V151max', 'V152max', 'V153max', 'V154max', 'V155max',
 'V156max', 'V157max', 'V158max', 'V159max', 'V160max', 'V161max', 'V162max',
 'V163max', 'V164max', 'V165max', 'V166max', 'V167max', 'V168max', 'V169max',
 'V170max', 'V171max', 'V172max', 'V173max', 'V174max', 'V175max', 'V176max',
 'V177max', 'V178max', 'V179max', 'V180max', 'V181max', 'V182max', 'V183max',
 'V184max', 'V185max', 'V186max', 'V187max', 'V188max', 'V189max', 'V190max',
 'V191max', 'V192max', 'V193max', 'V194max', 'V195max', 'V196max', 'V197max',
 'V198max', 'V199max', 'V200max', 'V201max', 'V202max', 'V203max', 'V204max',
 'V205max', 'V206max', 'V207max', 'V208max', 'V209max', 'V210max', 'V211max',
 'V212max', 'V213max', 'V214max', 'V215max', 'V216max', 'V217max', 'V218max',
 'V219max', 'V220max', 'V221max', 'V222max', 'V223max', 'V224max', 'V225max',
 'V226max', 'V227max', 'V228max', 'V229max', 'V230max', 'V231max', 'V232max',
 'V233max', 'V234max', 'V235max', 'V236max', 'V237max', 'V238max', 'V239max',
 'V240max', 'V241max', 'V242max', 'V243max', 'V244max', 'V245max', 'V246max',
 'V247max', 'V248max', 'V249max', 'V250max', 'V251max', 'V252max', 'V253max',
 'V254max', 'V255max', 'V256max', 'V257max', 'V258max', 'V259max', 'V260max',
 'V261max', 'V262max', 'V263max', 'V264max', 'V265max', 'V266max', 'V267max',
 'V268max', 'V269max', 'V270max', 'V271max', 'V272max', 'V273max', 'V274max',
 'V275max', 'V276max', 'V277max', 'V278max', 'V279max', 'V280max', 'V281max',
 'V282max', 'V283max', 'V284max', 'V285max', 'V286max', 'V287max', 'V288max',
 'V289max', 'V290max', 'V291max', 'V292max', 'V293max', 'V294max', 'V295max',
 'V296max', 'V297max', 'V298max', 'V299max', 'V300max', 'V301max', 'V302max',
 'V303max', 'V304max', 'V305max', 'V306max', 'V307max', 'V308max', 'V309max',
 'V310max', 'V311max', 'V312max', 'V313max', 'V314max', 'V315max', 'V316max',
 'V317max', 'V318max', 'V319max', 'V320max', 'V321max', 'V322max', 'V323max',
 'V324max', 'V325max', 'V326max', 'V327max', 'V328max', 'V329max', 'V330max',
 'V331max', 'V332max', 'V333max', 'V334max', 'V335max', 'V336max', 'V337max',
 'V338max', 'V339max', 'ntrans', 'min_amt', 'mean_amt', 'max_amt',
 'num_trans_ints', 'minC1', 'minC2', 'minC3', 'minC4', 'minC5', 'minC6',
 'minC7', 'minC8', 'minC9', 'minC10', 'minC11', 'minC12', 'minC13', 'minC14',
 'maxC1', 'maxC2', 'maxC3', 'maxC4', 'maxC5', 'maxC6', 'maxC7', 'maxC8',
 'maxC9', 'maxC10', 'maxC11', 'maxC12', 'maxC13', 'maxC14', 'countC1_inc',
 'countC2_inc', 'countC3_inc', 'countC4_inc', 'countC5_inc', 'countC6_inc',
 'countC7_inc', 'countC8_inc', 'countC9_inc', 'countC10_inc', 'countC11_inc',
 'countC12_inc', 'countC13_inc', 'countC14_inc', 'ndistM1', 'ndistM2',
 'ndistM3', 'ndistM4', 'ndistM5', 'ndistM6', 'ndistM7', 'ndistM8', 'ndistM9']

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

CAT_FEATURES = [c for c in CAT_FEATURES if c in FEATURES]

X = train_df[FEATURES].copy()
y = train_df[TARGET].copy()
X_test = test_df[FEATURES].copy()

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
# lgb_params = {
#     'min_data_in_leaf': 106,
#     'num_leaves': 500,
#     'learning_rate': LEARNING_RATE, #0.008,
#     'min_child_weight': 0.03454472573214212,
#     'bagging_fraction': 0.4181193142567742,
#     'feature_fraction': 0.3797454081646243,
#     'reg_lambda': 0.6485237330340494,
#     'reg_alpha': 0.3899927210061127,
#     'max_depth': DEPTH, #-1,
#     'objective': 'binary',
#     'seed': RANDOM_STATE, #13,
#     'feature_fraction_seed': RANDOM_STATE, #13,
#     'bagging_seed': RANDOM_STATE, #13,
#     'drop_seed': RANDOM_STATE, #13,
#     'data_random_seed': RANDOM_STATE, #13,
#     'boosting_type': 'gbdt',
#     'verbose': 1,
#     'metric':'auc',
#     'n_estimators':N_ESTIMATORS,
# }


def train_lightgbm(X_train, y_train, X_valid, y_valid, X_test, CAT_FEATURES, fold_n, feature_importance):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    if len(CAT_FEATURES) > 0:
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

del train_df, test_df
gc.collect()

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
