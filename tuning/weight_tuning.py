import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import pickle
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin_cobyla

start_time = datetime.now()

##################################################################################
# cal metric

def normalized_gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1.0 / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1.0 / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

##################################################################################
# pre-processing

def make_dummies(df, variables):
    for variable in variables:
        dummies = pd.get_dummies(df[variable], prefix = variable)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(variable, 1)
    return df


##################################################################################
# load data

build_feat = False

if build_feat is True:
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
    train.drop('T1_V6', axis=1, inplace=True)

    test.drop('T2_V10', axis=1, inplace=True)
    test.drop('T2_V7', axis=1, inplace=True)
    test.drop('T1_V13', axis=1, inplace=True)
    test.drop('T1_V10', axis=1, inplace=True)
    test.drop('T1_V6', axis=1, inplace=True)

    # columns and index for later use
    columns = train.columns
    test_ind = test.index

    # into array
    arr_train = np.array(train)
    arr_test = np.array(test)

    # cat -> numeric
    for i in range(arr_train.shape[1]):
        if type(arr_train[1,i]) is str:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(arr_train[:,i]) + list(arr_test[:,i]))
            arr_train[:,i] = lbl.transform(arr_train[:,i])
            arr_test[:,i] = lbl.transform(arr_test[:,i])

    train_numeric = arr_train.astype(np.int64)
    test_numeric = arr_test.astype(np.int64)

    # cat -> numeric, with ordering
    train_numeric_order = pd.DataFrame(train, copy=True)
    test_numeric_order = pd.DataFrame(test, copy=True)

    df_combine = pd.concat([train_numeric_order, test_numeric_order])

    change = {}
    for key in df_combine.select_dtypes(include=['object']).columns.tolist():
        sub_change = df_combine[key].value_counts().index.tolist()
        change[key] = {sub_change[k]:k for k in range(len(sub_change))}

    for key in df_combine.select_dtypes(include=['object']).columns.tolist():
        train_numeric_order[key] = train_numeric_order[key].map(change[key])
        test_numeric_order[key] = test_numeric_order[key].map(change[key])

    train_numeric_order = np.array(train_numeric_order).astype(np.int64)
    test_numeric_order = np.array(test_numeric_order).astype(np.int64)

    # onehot encode
    ohc = preprocessing.OneHotEncoder()
    ohc.fit(np.vstack((train_numeric,test_numeric)))
    train_onehot = ohc.transform(train_numeric)
    test_onehot = ohc.transform(test_numeric)

    cats = ['T1_V4', 'T1_V5', 'T2_V13', 'T1_V7', 'T1_V8', 
            'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16',
            'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',]

    train.loc[:, cats] = train[cats].applymap(str)
    test.loc[:, cats] = test[cats].applymap(str)
    train = make_dummies(train, cats)
    test = make_dummies(test, cats)

    train_c2o = np.array(train).astype(float)
    test_c2o = np.array(test).astype(float)
    train_numeric = train_numeric.astype(float)
    test_numeric = test_numeric.astype(float)
    train_numeric_order = train_numeric_order.astype(float)
    test_numeric_order = test_numeric_order.astype(float)
    train_relate = np.array(train_numeric)
    test_relate = np.array(test_numeric)

    train_log_order = np.array(train_numeric_order)
    test_log_order = np.array(test_numeric_order)
    train_log_order = np.log(train_log_order + 1)
    test_log_order = np.log(test_log_order + 1)

    train_log = np.array(train_numeric)
    test_log = np.array(test_numeric)
    train_log = np.log(train_log + 1)
    test_log = np.log(test_log + 1)


    # add variable interactions to numeric and c2o data set
    train_relate = np.hstack((train_relate, np.array(train['T1_V1']*train['T1_V2']).reshape((len(train_numeric),1))))
    train_relate = np.hstack((train_relate, np.array(train['T1_V1']*train['T1_V3']).reshape((len(train_numeric),1))))
    train_relate = np.hstack((train_relate, np.array(train['T1_V2']*train['T1_V3']).reshape((len(train_numeric),1))))
    train_relate = np.hstack((train_relate, np.array(train['T2_V1']/train['T1_V1']).reshape((len(train_numeric),1))))
    train_relate = np.hstack((train_relate, np.array(train['T2_V1']/train['T1_V2']).reshape((len(train_numeric),1))))
    train_relate = np.hstack((train_relate, np.array(train['T2_V1']*train['T2_V2']).reshape((len(train_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T1_V1']*test['T1_V2']).reshape((len(test_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T1_V1']*test['T1_V3']).reshape((len(test_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T1_V2']*test['T1_V3']).reshape((len(test_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T2_V1']/test['T1_V1']).reshape((len(test_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T2_V1']/test['T1_V2']).reshape((len(test_numeric),1))))
    test_relate = np.hstack((test_relate, np.array(test['T2_V1']*test['T2_V2']).reshape((len(test_numeric),1))))

    feat_pack = (train_numeric, test_numeric, train_onehot, test_onehot, train_c2o, test_c2o, train_numeric_order, test_numeric_order, train_relate, test_relate, train_log_order, test_log_order, train_log, test_log, train_y, test_ind)
    joblib.dump(feat_pack, './data/feat_pack.pkl', compress = 3)

else:
    train_numeric, test_numeric, train_onehot, test_onehot, train_c2o, test_c2o, train_numeric_order, test_numeric_order, train_relate, test_relate, train_log_order, test_log_order, train_log, test_log, train_y, test_ind = joblib.load('./data/feat_pack.pkl')


original_y = np.array(train_y)
indicator = 1./1.6
transformed_y = np.power(np.array(train_y), indicator)
log_y = np.log(np.array(train_y))


##################################################################################
# generate 10-fold data
##################################################################################

# kf = KFold(n=len(train_y), n_folds=5, shuffle=True, random_state=42)

# joblib.dump(kf, './data/tuning_kf.pkl', compress = 3)

kf = joblib.load('./data/tuning_kf.pkl')

##################################################################################
# target and predict for each fold
##################################################################################

# record predictions and target

# print 'generating target ...'
# target = []
# for train_idx, test_idx in kf:
#     # the target
#     target = np.append(target, original_y[test_idx])

# joblib.dump(target, './data/tuning_target.pkl', compress = 3)

# print 'generating transformed target ...'
# transformed_target = []
# for train_idx, test_idx in kf:
#     # the target
#     transformed_target = np.append(transformed_target, transformed_y[test_idx])

# joblib.dump(transformed_target, './data/tuning_transformed_target.pkl', compress = 3)


# print 'generating rf log_y...'
# rf_log_y = []
# for train_idx, test_idx in kf:
#     train_data = train_numeric[train_idx]
#     train_target = log_y[train_idx]
#     test_data = train_numeric[test_idx]
#     # the target
#     reg_1 = RandomForestRegressor(n_estimators=500, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
#     reg_1.fit(train_data, train_target)
#     pred_1 = reg_1.predict(test_data)
#     rf_log_y = np.append(rf_log_y, pred_1)

# joblib.dump(rf_log_y, './data/tuning_rf_log_y.pkl', compress = 3)

# print 'generating rf order ...'
# rf_order = []
# for train_idx, test_idx in kf:
#     train_data = train_numeric_order[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_numeric_order[test_idx]
#     # the target
#     reg_15 = RandomForestRegressor(n_estimators=500, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
#     reg_15.fit(train_data, train_target)
#     pred_15 = reg_15.predict(test_data)
#     rf_order = np.append(rf_order, pred_15)

# joblib.dump(rf_order, './data/tuning_rf_order.pkl', compress = 3)


# print 'generating rf relate ...'
# rf_relate = []
# for train_idx, test_idx in kf:
#     train_data = train_relate[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_relate[test_idx]
#     # the target
#     reg_16 = RandomForestRegressor(n_estimators=500, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
#     reg_16.fit(train_data, train_target)
#     pred_16 = reg_16.predict(test_data)
#     rf_relate = np.append(rf_relate, pred_16)

# joblib.dump(rf_relate, './data/tuning_rf_relate.pkl', compress = 3)

# print 'generating rf c2o ...'
# rf_c2o = []
# for train_idx, test_idx in kf:
#     train_data = train_c2o[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_c2o[test_idx]
#     # the target
#     reg_17 = RandomForestRegressor(n_estimators=500, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
#     reg_17.fit(train_data, train_target)
#     pred_17 = reg_17.predict(test_data)
#     rf_c2o = np.append(rf_c2o, pred_17)

# joblib.dump(rf_c2o, './data/tuning_rf_c2o.pkl', compress = 3)

# print 'generating rf log (order) ...'
# rf_log = []
# for train_idx, test_idx in kf:
#     train_data = train_log[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_log[test_idx]
#     # the target
#     reg_21 = RandomForestRegressor(n_estimators=500, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
#     reg_21.fit(train_data, train_target)
#     pred_21 = reg_21.predict(test_data)
#     rf_log = np.append(rf_log, pred_21)

# joblib.dump(rf_log, './data/tuning_rf_log.pkl', compress = 3)


# # para for xgb
# params = {}
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["max_depth"] = 7
# params["subsample"] = 0.65
# params["colsample_bytree"] = 0.4
# params["min_child_weight"] = 5
# params["silent"] = 0
# # params["scale_pos_weight"] = 1
# # plst = list(params.items())
# num_rounds = 100000


# print 'generating xgb on numeric ...'
# xgb_numeric = np.zeros(len(transformed_y))

# for i in range(1):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_numeric[train_idx]
#         test_data = train_numeric[test_idx]
#         train_target = transformed_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_2 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_2 = reg_2.predict(xgtest, ntree_limit=reg_2.best_iteration)
#         xgb_1round = np.append(xgb_1round, pred_2)

#     xgb_numeric += xgb_1round
# xgb_numeric = xgb_numeric/1

# joblib.dump(xgb_numeric, './data/tuning_10_xgb_numeric.pkl', compress = 3)


# print 'generating xgb on cat to onehot with log_y ...'
# xgb_c2o_log_y = np.zeros(len(log_y))

# for i in range(2):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_c2o[train_idx]
#         test_data = train_c2o[test_idx]
#         train_target = log_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_3 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_3 = reg_3.predict(xgtest, ntree_limit=reg_3.best_iteration)

#         xgb_1round = np.append(xgb_1round, pred_3)

#     xgb_c2o_log_y += xgb_1round
# xgb_c2o_log_y = xgb_c2o_log_y/2

# joblib.dump(xgb_c2o_log_y, './data/tuning_10_xgb_c2o_log_y.pkl', compress = 3)


# print 'generating xgb on ordered numeric ...'

# xgb_numeric_order = np.zeros(len(transformed_y))

# for i in range(2):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_numeric_order[train_idx]
#         test_data = train_numeric_order[test_idx]
#         train_target = transformed_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_4 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_4 = reg_4.predict(xgtest, ntree_limit=reg_4.best_iteration)
#         xgb_1round = np.append(xgb_1round, pred_4)

#     xgb_numeric_order += xgb_1round
# xgb_numeric_order = xgb_numeric_order/2

# joblib.dump(xgb_numeric_order, './data/tuning_10_xgb_numeric_order.pkl', compress = 3)


# print 'generating xgb on numeric with relation with log _y ...'
# xgb_relate_log_y = np.zeros(len(log_y))

# for i in range(1):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_relate[train_idx]
#         test_data = train_relate[test_idx]
#         train_target = log_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_5 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_5 = reg_5.predict(xgtest, ntree_limit=reg_5.best_iteration)

#         xgb_1round = np.append(xgb_1round, pred_5)

#     xgb_relate_log_y += xgb_1round
# xgb_relate_log_y = xgb_relate_log_y/1

# joblib.dump(xgb_relate_log_y, './data/tuning_10_xgb_relate_log_y.pkl', compress = 3)

# print 'generating xgb on log ...'
# xgb_log_ordered = np.zeros(len(transformed_y))

# for i in range(1):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_log_order[train_idx]
#         test_data = train_log_order[test_idx]
#         train_target = transformed_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_20 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_20 = reg_20.predict(xgtest, ntree_limit=reg_20.best_iteration)

#         xgb_1round = np.append(xgb_1round, pred_20)

#     xgb_log_ordered += xgb_1round
# xgb_log_ordered = xgb_log_ordered/1

# joblib.dump(xgb_log_ordered, './data/tuning_10_xgb_log_order.pkl', compress = 3)

# print 'generating xgb on log with log y ...'
# xgb_log_no_ordered_log_y = np.zeros(len(log_y))

# for i in range(1):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_log[train_idx]
#         test_data = train_log[test_idx]
#         train_target = log_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_20 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_20 = reg_20.predict(xgtest, ntree_limit=reg_20.best_iteration)

#         xgb_1round = np.append(xgb_1round, pred_20)

#     xgb_log_no_ordered_log_y += xgb_1round
# xgb_log_no_ordered_log_y = xgb_log_no_ordered_log_y/1

# joblib.dump(xgb_log_no_ordered_log_y, './data/tuning_10_xgb_log_no_order_log_y.pkl', compress = 3)


# print 'generating xgb on numeric and log y ...'
# xgb_numeric_log_y = np.zeros(len(log_y))

# for i in range(1):

#     params["seed"] = i
#     plst = list(params.items())

#     xgb_1round = []
#     for train_idx, test_idx in kf:
#         train_data = train_numeric[train_idx]
#         test_data = train_numeric[test_idx]
#         train_target = log_y[train_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgval, 'val')]
#         reg_222 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         pred_222 = reg_222.predict(xgtest, ntree_limit=reg_222.best_iteration)
#         xgb_1round = np.append(xgb_1round, pred_222)

#     xgb_numeric_log_y += xgb_1round
# xgb_numeric_log_y = xgb_numeric_log_y/1

# joblib.dump(xgb_numeric_log_y, './data/tuning_10_xgb_numeric_log_y.pkl', compress = 3)



print 'generating lasso on onehot ...'
lasso_onehot = []
for train_idx, test_idx in kf:
    train_data = train_onehot[train_idx]
    train_target = transformed_y[train_idx]
    test_data = train_onehot[test_idx]
    # the target
    reg_4 = Lasso(alpha=0.00035)
    reg_4.fit(train_data, train_target)
    pred_4 = reg_4.predict(test_data)
    lasso_onehot = np.append(lasso_onehot, pred_4)
joblib.dump(lasso_onehot, './data/tuning_lasso_onehot.pkl', compress = 3)

print 'generating ridge on onehot ...'
ridge_onehot = []
for train_idx, test_idx in kf:
    train_data = train_onehot[train_idx]
    train_target = transformed_y[train_idx]
    test_data = train_onehot[test_idx]
    # the target
    reg_5 = Ridge(alpha=120)
    reg_5.fit(train_data, train_target)
    pred_5 = reg_5.predict(test_data)
    ridge_onehot = np.append(ridge_onehot, pred_5)
joblib.dump(ridge_onehot, './data/tuning_ridge_onehot.pkl', compress = 3)


##################################################################################
# use colyba to min func
##################################################################################

# result = pd.DataFrame()
# result['target'] = target
# result['transformed_target'] = transformed_target
# result['rf'] = rf
# result['xgb_numeric'] = xgb_numeric
# result['xgb_c2o'] = xgb_c2o
# result['xgb_numeric_order'] = xgb_numeric_order
# result['xgb_relate'] = xgb_relate
# # result['lasso'] = lasso_onehot
# # result['ridge'] = ridge_onehot
# result.to_csv('./info/result_addition.csv')


# predictions = np.hstack((rf.reshape(len(rf),1),
#                          xgb_numeric.reshape(len(xgb_numeric),1),
#                          xgb_c2o.reshape(len(xgb_c2o),1),
#                          xgb_numeric_order.reshape(len(xgb_numeric_order),1),
#                          xgb_relate.reshape(len(xgb_relate),1)))

# # weights are non-negative
# def constraint(w, *args):
#     return min(w) - .0

# # maximize gini score is min -score
# def gini_score(w, predictions, target):
#     preds = np.dot(predictions, w)
#     score = normalized_gini(target, preds)
#     return 0. - score

# # minimize rmse
# def mse_score(w, predictions, target):
#     preds = np.dot(predictions, w)
#     preds /= sum(w)
#     score = mean_squared_error(target, preds)
#     return score

# # initial weights
# print 'calculating gini weights ...'
# w0 = [1.0] * predictions.shape[1]
# weights_gini = fmin_cobyla(gini_score, w0, args=(predictions, target), cons=[constraint], rhoend=1e-5)

# print normalized_gini(target, np.dot(predictions, w0))

# # initial weights
# print 'calculating rmse weights ...'
# w0 = [1.0] * predictions.shape[1]
# weights_mse = fmin_cobyla(mse_score, w0, args=(predictions, transformed_target), cons=[constraint], rhoend=1e-5)


# print weights
# print weights / np.sum(weights)

# # gini score
# # array([  5.22870163e-02,   2.07893145e-01,   4.20693372e+00,
# #          4.75759443e+00,  -6.01853108e-36,   3.37402175e-04])
# # mse
# # array([  4.04695626e-01,  -1.11759737e-05,   4.02053983e+00,
# #          4.50227578e+00,   4.40458002e-01,  -1.50774184e-06])

# 0.625 one but it's wrong (cry~~~)
# tuning.weights_gini
# Out[214]: 
# array([  4.51160132e+00,   8.23478073e-04,   9.64230134e-03,
#          8.47569939e-01,   1.38333377e+00])

# tuning.weights_mse
# Out[215]: 
# array([  3.95752996e+00,  -6.48545959e-05,  -7.65708709e-05,
#          2.40754840e+00,   2.69471332e+00])

# choose: rf, gb_numeric, gb_onehot
# weights_gini
# Out[311]: 
# array([  1.90470718e-01,   1.83212042e+00,   1.84226528e+00,
#          2.77555756e-21])

# weight gini, 0.38853393565214117
# array([ 0.75649167,  3.20492053,  3.08214178,  2.85790244,  2.31080358])

# weights_gini, add rf_order(6 in total), 0.388535620935
# Out[385]: 
# array([  6.17127419e-01,   2.88456469e+00,   2.73163919e+00,
#          2.35426748e+00,   2.16003878e+00,   6.89278122e-04])


# weights_gini
# 0.388597087075 -----
# rf, xgb_numeric, xgb_c2o, xgb_numeric_order, xgb_relate, rf_order, rf_relate
# array([  2.06944409e-02,   2.92116748e+00,   3.16930094e+00,
#          2.12913000e+00,   1.59987511e+00,   4.16949868e-04,
#          8.00732786e-01])

# 0.38867167
# array([True,False,True,False,True,True,True,False,False,False,True,True])]
# 
# rf, xgb_c2o, xgb_relate, rf_order, rf_relate, xgb_log_no_order, xgb_log_order
# 
# weights_gini
# Out[533]: 
# array([  2.51002426e-02,   2.92015566e+00,   1.35444296e+00,
#          2.53197895e-04,   7.14555690e-01,   2.75396878e+00,
#          2.25504823e+00])

# predictions = hstack((rf.reshape(len(rf),1),
#                       xgb_c2o.reshape(len(xgb_c2o),1),
#                       xgb_relate.reshape(len(xgb_relate),1),
#                       rf_order.reshape(len(rf_order),1),
#                       rf_relate.reshape(len(rf_relate),1),
#                       xgb_log_no_order.reshape(len(xgb_log_no_order),1),
#                       xgb_log_order.reshape(len(xgb_log_order),1),
#                       rf_log_y.reshape(len(rf_log_y),1),
#                       xgb_numeric_log_y.reshape(len(xgb_numeric_log_y),1),
#                       xgb_c2o_log_y.reshape(len(xgb_c2o_log_y),1),
#                       xgb_relate_log_y.reshape(len(xgb_relate_log_y),1),---no need
#                       xgb_log_order_log_y.reshape(len(xgb_log_order_log_y),1),
#                       xgb_log_no_order_log_y.reshape(len(xgb_log_no_order_log_y),1)))

# weights_gini
# Out[674]: 
# array([  4.67580555e-01,   2.11711896e+00,   1.70813157e+00,
#          6.05814076e-05,   5.69982450e-01,   1.79998417e+00,
#          1.16805668e+00,   2.64736293e-02,   7.27634375e-01,
#          2.60181138e+00,   1.56868769e-04,   1.66585218e+00])
