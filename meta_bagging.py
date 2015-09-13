'''
Do n iterations:

1.first level
- choose split ratio
- train models on level-1
- combine prediction and raw
2.second level
- train models on level-2

'''

import pandas as pd
import numpy as np
import pickle
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from scipy.optimize import fmin_cobyla
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from datetime import datetime

script_start_time = datetime.now()

##################################################################################
# load raw train and test data
##################################################################################

train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)
train_y = np.array(train.Hazard).astype(float)

# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)
# drop noisy features
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)

# columns and index for later use
columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

train_raw = train.astype(float)
test_raw = test.astype(float)

##################################################################################
#load one-hot train and test data
##################################################################################

with open('./data/train_denoise.vec', 'rb') as f:
    train_onehot = pickle.load(f)

with open('./data/test_denoise.vec', 'rb') as f:
    test_onehot = pickle.load(f)

##################################################################################
# load train and test 
##################################################################################

pred_sum = None
iter_round = 20 # num of bagging rounds
l1_ratio = 0.5 # define l1 / l1+l2 split ratio


for i in range(iter_round):

    loop_start_time = datetime.now()

    print '##################################################################################'
    print 'round {}'.format(i)
    print '##################################################################################'
    # level 1's training
    l1, l2= train_test_split(range(train.shape[0]), train_size=l1_ratio, random_state=i)

    # rf on raw
    print 'generating rf ...'
    rgrs_1 = RandomForestRegressor(n_estimators=500, max_features=10, max_depth=15, min_samples_leaf=4, n_jobs=-1)
    rgrs_1.fit(train_raw[l1], train_y[l1])
    pred_1 = rgrs_1.predict(train_raw[l2])
    pred_1_test = rgrs_1.predict(test_raw)


    print 'generating et ...'
    rgrs_2 = ExtraTreesRegressor(n_estimators=500, max_features=15, max_depth=15, min_samples_leaf=4, n_jobs=-1)
    rgrs_2.fit(train_raw[l1], train_y[l1])
    pred_2 = rgrs_2.predict(train_raw[l2])
    pred_2_test = rgrs_2.predict(test_raw)

    # xgb on raw
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["max_depth"] = 7
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["min_child_weight"] = 5
    params["silent"] = 1
    plst = list(params.items())

    eval_rat = int(0.8*len(l1))
    print 'generating xgb on raw ...'
    dtrain3 = xgb.DMatrix(train_raw[l1[:eval_rat]], train_y[l1[:eval_rat]])
    deval3 = xgb.DMatrix(train_raw[l1[eval_rat:]], train_y[l1[eval_rat:]])
    dtest3 = xgb.DMatrix(train_raw[l2])
    watchlist3 = [(dtrain3,'train'), (deval3,'eval')]
    rgrs_3 = xgb.train(plst, dtrain3, 10000, watchlist3, early_stopping_rounds=30)
    pred_3 = rgrs_3.predict(dtest3, ntree_limit=rgrs_3.best_iteration)
    pred_3_test = rgrs_3.predict(xgb.DMatrix(test_raw), ntree_limit=rgrs_3.best_iteration)

    # xgb on one-hot
    print 'generating xgb one hot ...'
    dtrain4 = xgb.DMatrix(train_onehot.iloc[l1[:eval_rat]], train_y[l1[:eval_rat]])
    deval4 = xgb.DMatrix(train_onehot.iloc[l1[eval_rat:]], train_y[l1[eval_rat:]])
    dtest4 = xgb.DMatrix(train_onehot.iloc[l2])
    watchlist4 = [(dtrain4,'train'), (deval4,'eval')]
    rgrs_4 = xgb.train(plst, dtrain4, 10000, watchlist4, early_stopping_rounds=30)
    pred_4 = rgrs_4.predict(dtest4, ntree_limit=rgrs_4.best_iteration)
    pred_4_test = rgrs_4.predict(xgb.DMatrix(test_onehot), ntree_limit=rgrs_4.best_iteration)

    # lasso on one-hot
    print 'generating lasso ...'
    rgrs_5 = Lasso(alpha=0.00135)
    rgrs_5.fit(train_onehot.iloc[l1], train_y[l1])
    pred_5 = rgrs_5.predict(train_onehot.iloc[l2])
    pred_5_test = rgrs_5.predict(test_onehot)

    # ridge on one-hot
    print 'generating ridge ...'
    rgrs_6 = Ridge(alpha=100)
    rgrs_6.fit(train_onehot.iloc[l1], train_y[l1])
    pred_6 = rgrs_6.predict(train_onehot.iloc[l2])
    pred_6_test = rgrs_6.predict(test_onehot)

    feat_pack = np.hstack((train_raw[l2], pred_1.reshape(len(pred_1),1), pred_2.reshape(len(pred_1),1), pred_3.reshape(len(pred_1),1), pred_4.reshape(len(pred_1),1), pred_5.reshape(len(pred_1),1), pred_6.reshape(len(pred_1),1)))
    test_pack = np.hstack((test_raw, pred_1_test.reshape(len(pred_1_test),1), pred_2_test.reshape(len(pred_1_test),1), pred_3_test.reshape(len(pred_1_test),1), pred_4_test.reshape(len(pred_1_test),1), pred_5_test.reshape(len(pred_1_test),1), pred_6_test.reshape(len(pred_1_test),1)))

    # level 2's training
    eval_rat_l2 = int(0.8*len(l2))
    dtrain = xgb.DMatrix(feat_pack[:eval_rat_l2], train_y[l2[:eval_rat_l2]])
    deval = xgb.DMatrix(feat_pack[eval_rat_l2:], train_y[l2[eval_rat_l2:]])
    dtest = xgb.DMatrix(test_pack)
    watchlist = [(dtrain,'train'), (deval,'eval')]
    rgrs = xgb.train(plst, dtrain, 10000, watchlist, early_stopping_rounds=30)
    pred = rgrs.predict(dtest, ntree_limit=rgrs.best_iteration)

    pred_sum = pred_sum + pred if pred_sum is not None else pred

    loop_end_time = datetime.now()

    print '##################################################################################'
    print '---- {}th loop duration is {}'.format(i, loop_end_time-loop_start_time)


pred = pred_sum / iter_round






with open('./data/meta_20round_1.bst', 'wb') as f:
    pickle.dump(pred, f)

preds = pd.DataFrame({"Id": test_ind, "Hazard": pred})
preds = preds.set_index('Id')
preds.to_csv('meta_20round_output.csv')








print '---- script duration is {}'.format(datetime.now()-script_start_time)


# duration 1:21:57 for 20 rounds