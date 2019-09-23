"""
Created by: Rob Mulla
Sep 21

IEEE Fraud Detection Model

- FE009
- Adding raddar user level features
- Add first, second, third digit of addr1 and addr2 features
- Drop only DOY features with low importance
- Different Parameters
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
FE_SET = 'FE011' # Feature Engineering Version

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

FEATURES = ['V85',
         'bank_type_TransactionAmt_mean',
         'D5_fq_enc',
         'V12',
         'V81',
         'V282',
         'bank_type_D7_std',
         'id_15',
         'V13',
         'C12_fq_enc',
         'anomaly',
         'D7_DT_D_std_score',
         'D3_DT_D_min_max',
         'card4_count_full',
         'D14_DT_D_min_max',
         'card1_count_full',
         'V169',
         'D3_DT_M_min_max',
         'V279',
         'V91',
         'bank_type_D10_std',
         'D14',
         'D6_DT_M_std_score',
         'D4_DT_W_min_max',
         'V152',
         'V56',
         'D3_intercept_bin0',
         'D14_intercept_bin0',
         'V220',
         'V277',
         'D12_intercept',
         'ProductCD_W_00cents',
         'D13_intercept_bin0',
         'V291',
         'V189',
         'D15_DT_M_min_max',
         'C5_fq_enc',
         'D3_fq_enc',
         'card5_fq_enc',
         'addr1_count_full',
         'V266',
         'D11_intercept_bin2',
         'V23',
         'D4_intercept_bin3',
         'bank_type_D10_mean',
         'D2_intercept_bin3',
         'V306',
         'DeviceType',
         'V285',
         'D5_DT_W_std_score',
         'V131',
         'V37',
         'V296',
         'bank_type_D1_mean',
         'V75',
         'D3_DT_W_std_score',
         'D10_DT_M_min_max',
         'id_33_0',
         'V67',
         'D4_intercept_bin4',
         'V256',
         'V143',
         'uid5_D6_std',
         'ProductCD_target_mean',
         'mxC3',
         'V129',
         'D13_DT_M_std_score',
         'V24',
         'D3_DT_M_std_score',
         'mxC4',
         'D9',
         'id_30_version_fq_enc',
         'D5_DT_D_std_score',
         'D11_DT_M_std_score',
         'uid5_D6_mean',
         'D14_DT_M_std_score',
         'card5_TransactionAmt_std',
         'V20',
         'C8_fq_enc',
         'V70',
         'V127',
         'D6_intercept',
         'D15_DT_W_min_max',
         'sum_Cxx_binary_higher_than_q95',
         'V156',
         'uid4_D12_mean',
         'C5',
         'uid4_D12_std',
         'id_30_fq_enc',
         'V61',
         'id_33',
         'D15_to_std_addr1',
         'bank_type_D9_mean',
         'D5_intercept',
         'D10_DT_W_min_max',
         'V130',
         'bank_type_D9_std',
         'uid5_D7_std',
         'bank_type_D14_mean',
         'bank_type_D3_std',
         'bank_type_D5_mean',
         'ProductCD',
         'M8',
         'V44',
         'D6_fq_enc',
         'D15_DT_D_min_max',
         'D11_intercept_bin0',
         'V257',
         'bank_type_D7_mean',
         'V76',
         'D15',
         'V38',
         'V55',
         'V261',
         'V149',
         'D4',
         'D8_intercept_bin0',
         'M2',
         'bank_type_D6_std',
         'id_30_version',
         'D4_intercept_bin1',
         'D15_to_mean_card4',
         'V82',
         'D3_DT_D_std_score',
         'D10_intercept_bin3',
         'bank_type_D2_std',
         'V77',
         'M7',
         'D11',
         'D4_intercept_bin2',
         'email_check',
         'V294',
         'V317',
         'V308',
         'id_33_fq_enc',
         'bank_type_D5_std',
         'D8_intercept',
         'V62',
         'V187',
         'card5_TransactionAmt_mean',
         'bank_type_D12_mean',
         'id_33_count_dist',
         'D2_intercept_bin2',
         'C10',
         'V86',
         'D8_DT_M_min_max',
         'D15_intercept_bin4',
         'D6_DT_W_std_score',
         'uid5_D7_mean',
         'C9_fq_enc',
         'mxC10',
         'D14_DT_W_std_score',
         'card2_count_full',
         'V258',
         'bank_type_D14_std',
         'D10_intercept_bin4',
         'V83',
         'bank_type_D13_std',
         'D8_DT_W_min_max',
         'TransactionAmt',
         'V312',
         'D14_intercept',
         'id_33_1',
         'D15_intercept_bin2',
         'D12_DT_W_std_score',
         'V78',
         'D8_D9_decimal_dist',
         'M9',
         'V281',
         'bank_type_D12_std',
         'V54',
         'C9',
         'M4_target_mean',
         'sum_Cxx_binary_higher_than_q90',
         'D10_DT_D_min_max',
         'bank_type_D3_mean',
         'bank_type_D8_mean',
         'R_emaildomain_prefix',
         'bank_type_D6_mean',
         'V314',
         'D11_DT_W_std_score',
         'D10',
         'D4_DT_D_min_max',
         'V283',
         'D10_intercept_bin2',
         'D13_intercept',
         'D8_DT_D_min_max',
         'C2_fq_enc',
         'V165',
         'D1_intercept_bin4',
         'bank_type_D13_mean',
         'D3_intercept',
         'TransactionAmt_2Dec',
         'card3_div_Mean_D9_DOY',
         'C12',
         'D4_DT_M_std_score',
         'D2_intercept_bin1',
         'mxC8',
         'D2_fq_enc',
         'addr1_third_digit',
         'D4_fq_enc',
         'D1_fq_enc',
         'mxC12',
         'D8',
         'D10_intercept_bin1',
         'id_01',
         'id_09',
         'id_03',
         'addr1_second_digit',
         'D15_to_mean_addr1',
         'sum_Cxx_binary_higher_than_q80',
         'V53',
         'TransactionAmt_decimal',
         'card3_div_Mean_D6_DOY',
         'D15_intercept_bin3',
         'V45',
         'id_02_to_std_card4',
         'addr2_div_Mean_D10_DOY_productCD',
         'DeviceInfo_version',
         'DeviceInfo_device',
         'D1_intercept_bin3',
         'D11_intercept',
         'DeviceInfo_version_fq_enc',
         'C6',
         'uid5_D13_std',
         'TransactionAmt_DT_M_min_max',
         'dist2',
         'C8',
         'D15_intercept_bin1',
         'M3',
         'R_emaildomain_fq_enc',
         'DeviceInfo_device_fq_enc',
         'D6_DT_D_std_score',
         'sum_Cxx_binary_higher_than_q60',
         'D11__DeviceInfo',
         'TranAmt_div_Mean_D12_DOY_productCD',
         'D10_DT_M_std_score',
         'uid5_D13_mean',
         'mxC5',
         'id_30',
         'addr2_div_Mean_D4_DOY',
         'uid2_D12_std',
         'C11_fq_enc',
         'id_06',
         'uid2_D12_mean',
         'sum_Cxx_binary_higher_than_q70',
         'V310',
         'V307',
         'C6_fq_enc',
         'D8_fq_enc',
         'dist2_fq_enc',
         'D2_intercept_bin0',
         'addr1_div_Mean_D10_DOY_productCD',
         'addr1_div_Mean_D10_DOY',
         'addr1_div_Mean_D11_DOY',
         'uid2_D8_std',
         'id_02__id_20',
         'V313',
         'D4_intercept_bin0',
         'D11_DT_D_std_score',
         'Transaction_day_of_week',
         'card6_div_Mean_D3_DOY',
         'uid2_D1_std',
         'uid5_D11_mean',
         'uid_fq_enc',
         'D14_DT_D_std_score',
         'D12_DT_D_std_score',
         'id_02_to_mean_card4',
         'uid4_D13_std',
         'D1_intercept_bin1',
         'id_02_to_std_card1',
         'uid5_D11_std',
         'P_emaildomain_prefix',
         'DT_day',
         'D8_DT_M_std_score',
         'uid2_D1_mean',
         'TransactionAmt_to_mean_card4',
         'card5_div_Mean_D11_DOY',
         'D15_DT_M_std_score',
         'V87',
         'uid_D12_std',
         'id_31_device_fq_enc',
         'uid2_D11_mean',
         'card3_DT_W_week_day_dist_best',
         'uid5_D14_std',
         'uid2_D15_mean',
         'sum_Cxx_binary_higher_than_q50',
         'id_13',
         'card3_div_Mean_D11_DOY',
         'C11',
         'bank_type_DT_W_week_day_dist_best',
         'card4_div_Mean_D11_DOY',
         'addr1_div_Mean_D1_DOY',
         'uid2_D4_mean',
         'card2_div_Mean_D11_DOY',
         'C13_fq_enc',
         'uid4_D13_mean',
         'card5_DT_W_week_day_dist_best',
         'id_02',
         'uid5_D14_mean',
         'uid2_D10_mean',
         'id_01_count_dist',
         'D13_DT_W_std_score',
         'C2',
         'C14',
         'addr2_div_Mean_D10_DOY',
         'uid2_D11_std',
         'addr1_div_Mean_D1_DOY_productCD',
         'id_02_to_mean_card1',
         'dist1_fq_enc',
         'card1_div_Mean_D11_DOY',
         'D15_to_std_card1',
         'TransactionAmt_DT_M_std_score',
         'uid2_D6_std',
         'TransactionAmt_to_std_card4',
         'uid2_D15_std',
         'uid3_D8_std',
         'card6_div_Mean_D11_DOY',
         'TranAmt_div_Mean_D14_DOY',
         'card3_div_Mean_D14_DOY',
         'D2',
         'D1',
         'uid_D15_mean',
         'uid4_D6_std',
         'uid_D15_std',
         'D10_intercept_bin0',
         'DeviceInfo_fq_enc',
         'uid2_D13_std',
         'uid_D12_mean',
         'uid4_D6_mean',
         'uid_D1_std',
         'D1_intercept_bin2',
         'uid_D10_mean',
         'card2__id_20',
         'uid4_D7_std',
         'uid3_D13_std',
         'C14_fq_enc',
         'uid_D8_std',
         'uid3_D13_mean',
         'uid2_D4_std',
         'addr1_div_Mean_D4_DOY',
         'uid_D4_mean',
         'D4_DT_W_std_score',
         'addr2_div_Mean_D1_DOY_productCD',
         'uid_D11_mean',
         'D15_intercept_bin0',
         'uid2_D10_std',
         'uid_D13_std',
         'uid2_fq_enc',
         'uid2_D13_mean',
         'uid2_D2_mean',
         'D2_intercept',
         'uid_D11_std',
         'card2',
         'uid4_D14_std',
         'C_sum_after_clip75',
         'R_emaildomain',
         'dist1',
         'id_05',
         'uid_TransactionAmt_mean',
         'uid_D1_mean',
         'uid3_D1_std',
         'uid5_D8_std',
         'uid3_D6_std',
         'Transaction_hour_of_day',
         'uid4_D14_mean',
         'uid5_D10_std',
         'uid3_D10_std',
         'uid5_D1_std',
         'uid5_D15_std',
         'uid2_D7_mean',
         'uid3_D11_std',
         'uid4_D8_std',
         'D13_DT_D_std_score',
         'uid3_D11_mean',
         'uid2_D14_std',
         'uid2_D7_std',
         'uid2_D14_mean',
         'uid_D13_mean',
         'uid_D10_std',
         'uid2_D3_std',
         'uid_D6_std',
         'uid3_D15_std',
         'addr1_fq_enc',
         'id_31',
         'uid_TransactionAmt_std',
         'card1_div_Mean_D4_DOY_productCD',
         'uid2_TransactionAmt_mean',
         'C_sum_after_clip90',
         'uid2_TransactionAmt_std',
         'uid4_D7_mean',
         'uid2_D6_mean',
         'uid3_D15_mean',
         'D15_to_mean_card1',
         'uid5_D15_mean',
         'M4',
         'uid3_D7_std',
         'card2_div_Mean_D4_DOY',
         'card5_div_Mean_D4_DOY_productCD',
         'card5_div_Mean_D4_DOY',
         'D4_intercept',
         'uid_D4_std',
         'card6_div_Mean_D4_DOY_productCD',
         'card5__P_emaildomain',
         'card1_fq_enc',
         'uid5_D10_mean',
         'card1_div_Mean_D4_DOY',
         'C1',
         'M6',
         'uid2_D2_std',
         'P_emaildomain_fq_enc',
         'card1_TransactionAmt_mean',
         'uid3_D10_mean',
         'TransactionAmt_DT_W_min_max',
         'uid5_D4_std',
         'card1_div_Mean_D10_DOY_productCD',
         'uid3_D1_mean',
         'card1_div_Mean_D10_DOY',
         'uid_D14_mean',
         'mxC9',
         'TranAmt_div_Mean_D4_DOY_productCD',
         'D15_DT_W_std_score',
         'DeviceInfo__P_emaildomain',
         'uid3_D14_mean',
         'bank_type_DT_M',
         'mxC11',
         'uid5_D1_mean',
         'uid_D2_mean',
         'D10_DT_W_std_score',
         'card3_DT_M_month_day_dist_best',
         'uid3_D2_std',
         'TranAmt_div_Mean_D4_DOY',
         'card1_TransactionAmt_std',
         'card3_div_Mean_D4_DOY_productCD',
         'D1_intercept_bin0',
         'uid3_D4_std',
         'card2_div_Mean_D10_DOY',
         'uid_D2_std',
         'uid3_D14_std',
         'uid3_D4_mean',
         'uid_D7_mean',
         'uid5_D2_std',
         'card4_div_Mean_D4_DOY_productCD',
         'card6_div_Mean_D4_DOY',
         'TranAmt_div_Mean_D10_DOY',
         'uid2_D9_std',
         'TransactionAmt_DT_W_std_score',
         'C1_fq_enc',
         'card1_div_Mean_D1_DOY',
         'uid5_D4_mean',
         'uid3_D6_mean',
         'mxC14',
         'uid5_D2_mean',
         'card4_div_Mean_D4_DOY',
         'card3_div_Mean_D4_DOY',
         'uid_D14_std',
         'M5',
         'C13',
         'mxC6',
         'card5_div_Mean_D10_DOY_productCD',
         'card3_DT_M_month_day_dist',
         'card2_div_Mean_D10_DOY_productCD',
         'uid_D7_std',
         'card2_div_Mean_D4_DOY_productCD',
         'bank_type_DT_M_month_day_dist',
         'uid3_D7_mean',
         'uid_D3_std',
         'uid5_fq_enc',
         'uid3_fq_enc',
         'uid_D3_mean',
         'D4_DT_D_std_score',
         'uid3_D2_mean',
         'uid4_D1_std',
         'uid2_D5_std',
         'uid4_D10_std',
         'bank_type_DT_D_hour_dist_best',
         'uid2_D8_mean',
         'card6_div_Mean_D10_DOY_productCD',
         'card1_div_Mean_D1_DOY_productCD',
         'uid5_D9_std',
         'card4_div_Mean_D10_DOY_productCD',
         'uid2_D3_mean',
         'uid_D6_mean',
         'card2_div_Mean_D1_DOY',
         'card5_div_Mean_D10_DOY',
         'mxC2',
         'card2_TransactionAmt_std',
         'bank_type_DT_W_week_day_dist',
         'card2_TransactionAmt_mean',
         'uid4_D10_mean',
         'id_31_count_dist',
         'TranAmt_div_Mean_D1_DOY',
         'uid3_D3_std',
         'uid4_D15_std',
         'card5_div_Mean_D1_DOY_productCD',
         'card4_div_Mean_D10_DOY',
         'card5_DT_D_hour_dist_best',
         'uid4_D4_std',
         'card5_DT_M_month_day_dist',
         'bank_type_DT_W',
         'addr1__card1',
         'bank_type_DT_M_month_day_dist_best',
         'card2_div_Mean_D1_DOY_productCD',
         'card6_div_Mean_D10_DOY',
         'uid2_D5_mean',
         'uid_DT_M',
         'card2__dist1',
         'uid2_D9_mean',
         'card5_DT_M_month_day_dist_best',
         'TranAmt_div_Mean_D10_DOY_productCD',
         'uid4_D11_std',
         'uid_D5_mean',
         'uid5_D3_std',
         'TransactionAmt_DT_D_std_score',
         'D8_DT_W_std_score',
         'card5_DT_W_week_day_dist',
         'uid5_D5_std',
         'card3_DT_W_week_day_dist',
         'uid4_D9_std',
         'D10_intercept',
         'uid3_D3_mean',
         'uid4_D5_std',
         'uid_D5_std',
         'card5_div_Mean_D1_DOY',
         'uid5_D3_mean',
         'bank_type_DT_D',
         'uid4_D1_mean',
         'uid_D8_mean',
         'uid3_D5_mean',
         'D15_intercept',
         'uid5_TransactionAmt_std',
         'uid3_D5_std',
         'uid4_D4_mean',
         'uid4_D15_mean',
         'uid5_D8_mean',
         'uid5_D9_mean',
         'uid_D9_std',
         'uid_D9_mean',
         'uid5_D5_mean',
         'mtransamt',
         'bank_type_DT_D_hour_dist',
         'uid4_D11_mean',
         'D15_DT_D_std_score',
         'TransactionAmt_DT_D_min_max',
         'uid4_D2_mean',
         'ntrans',
         'addr2_div_Mean_D1_DOY',
         'uid5_TransactionAmt_mean',
         'uid3_D9_std',
         'TransactionAmt_Dec',
         'uid3_TransactionAmt_std',
         'card5_DT_D_hour_dist',
         'card1',
         'card4_div_Mean_D1_DOY_productCD',
         'P_emaildomain__C2',
         'card3_div_Mean_D10_DOY',
         'uid4_D3_std',
         'card3_DT_D_hour_dist_best',
         'uid4_D8_mean',
         'uid4_D2_std',
         'card6_div_Mean_D1_DOY_productCD',
         'uid_DT_W',
         'Sum_TransAmt_Day',
         'uid4_D5_mean',
         'card4_div_Mean_D1_DOY',
         'card3_div_Mean_D10_DOY_productCD',
         'uid3_D8_mean',
         'TransactionAmt_userid_median',
         'uid4_fq_enc',
         'uid3_TransactionAmt_mean',
         'uid3_D9_mean',
         'card6_div_Mean_D1_DOY',
         'Trans_Count_Day',
         'mxC1',
         'D10_DT_D_std_score',
         'card3_div_Mean_D1_DOY',
         'TransactionAmt_to_mean_card1',
         'card2_fq_enc',
         'product_type',
         'card3_div_Mean_D1_DOY_productCD',
         'TransactionAmt_to_std_card1',
         'uid_DT_D',
         'uid4_D9_mean',
         'D1_intercept',
         'card3_DT_D_hour_dist',
         'TranAmt_div_Mean_D1_DOY_productCD',
         'product_type_DT_M',
         'uid4_D3_mean',
         'uid4_TransactionAmt_mean',
         'uid4_TransactionAmt_std',
         'D8_DT_D_std_score',
         'Mean_TransAmt_Day',
         'minDT',
         'product_type_DT_W',
         'mintransamt',
         'maxtransamt',
         'TransactionAmt_userid_std',
         'P_emaildomain',
         'card1__card5',
         'product_type_DT_D',
         'mxC13',
         'maxDT',
         'id_19',
         'DeviceInfo',
         'id_20',
         'addr1',
         'userid_min_C1',
         'userid_max_C1',
         'userid_max_minus_min_C1',
         'userid_unique_C1',
         'userid_mean_C1',
         'userid_min_C2',
         'userid_max_C2',
         'userid_max_minus_min_C2',
         'userid_unique_C2',
         'userid_mean_C2',
         'userid_min_C3',
         'userid_max_C3',
         'userid_max_minus_min_C3',
         'userid_unique_C3',
         'userid_mean_C3',
         'userid_min_C4',
         'userid_max_C4',
         'userid_max_minus_min_C4',
         'userid_unique_C4',
         'userid_mean_C4',
         'userid_min_C5',
         'userid_max_C5',
         'userid_max_minus_min_C5',
         'userid_unique_C5',
         'userid_mean_C5',
         'userid_min_C6',
         'userid_max_C6',
         'userid_max_minus_min_C6',
         'userid_unique_C6',
         'userid_mean_C6',
         'userid_min_C7',
         'userid_max_C7',
         'userid_max_minus_min_C7',
         'userid_unique_C7',
         'userid_mean_C7',
         'userid_min_C8',
         'userid_max_C8',
         'userid_max_minus_min_C8',
         'userid_unique_C8',
         'userid_mean_C8',
         'userid_min_C9',
         'userid_max_C9',
         'userid_max_minus_min_C9',
         'userid_unique_C9',
         'userid_mean_C9',
         'userid_min_C10',
         'userid_max_C10',
         'userid_max_minus_min_C10',
         'userid_unique_C10',
         'userid_mean_C10',
         'userid_min_C11',
         'userid_max_C11',
         'userid_max_minus_min_C11',
         'userid_unique_C11',
         'userid_mean_C11',
         'userid_min_C12',
         'userid_max_C12',
         'userid_max_minus_min_C12',
         'userid_unique_C12',
         'userid_mean_C12',
         'userid_min_C13',
         'userid_max_C13',
         'userid_max_minus_min_C13',
         'userid_unique_C13',
         'userid_mean_C13',
         'userid_min_C14',
         'userid_max_C14',
         'userid_max_minus_min_C14',
         'userid_unique_C14',
         'userid_mean_C14']

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
