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

build_feat = True

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

    # # numeric label, with ordering from most frequent to least
    # train_numeric = pd.DataFrame(train, copy=True)
    # test_numeric = pd.DataFrame(test, copy=True)

    # df_combine = pd.concat([train_numeric, test_numeric])

    # change = {}
    # for key in df_combine.select_dtypes(include=['object']).columns.tolist():
    #     sub_change = df_combine[key].value_counts().index.tolist()
    #     change[key] = {sub_change[k]:k for k in range(len(sub_change))}

    # for key in df_combine.select_dtypes(include=['object']).columns.tolist():
    #     train_numeric[key] = train_numeric[key].map(change[key])
    #     test_numeric[key] = test_numeric[key].map(change[key])

    # train_numeric = np.array(train_numeric).astype(np.int64)
    # test_numeric = np.array(test_numeric).astype(np.int64)


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

    # # add variable interactions to numeric and c2o data set
    # train_c2o = np.hstack((train_c2o, np.array(train['T1_V1']*train['T1_V2']).reshape((len(train_numeric),1))))
    # train_c2o = np.hstack((train_c2o, np.array(train['T1_V1']*train['T1_V3']).reshape((len(train_numeric),1))))
    # train_c2o = np.hstack((train_c2o, np.array(train['T1_V2']*train['T1_V3']).reshape((len(train_numeric),1))))
    # train_c2o = np.hstack((train_c2o, np.array(train['T2_V1']/train['T1_V1']).reshape((len(train_numeric),1))))
    # train_c2o = np.hstack((train_c2o, np.array(train['T2_V1']/train['T1_V2']).reshape((len(train_numeric),1))))
    # train_c2o = np.hstack((train_c2o, np.array(train['T2_V1']*train['T2_V2']).reshape((len(train_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T1_V1']*test['T1_V2']).reshape((len(test_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T1_V1']*test['T1_V3']).reshape((len(test_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T1_V2']*test['T1_V3']).reshape((len(test_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T2_V1']/test['T1_V1']).reshape((len(test_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T2_V1']/test['T1_V2']).reshape((len(test_numeric),1))))
    # test_c2o = np.hstack((test_c2o, np.array(test['T2_V1']*test['T2_V2']).reshape((len(test_numeric),1))))

    # train_numeric = np.hstack((train_numeric, np.array(train['T1_V1']*train['T1_V2']).reshape((len(train_numeric),1))))
    # train_numeric = np.hstack((train_numeric, np.array(train['T1_V1']*train['T1_V3']).reshape((len(train_numeric),1))))
    # train_numeric = np.hstack((train_numeric, np.array(train['T1_V2']*train['T1_V3']).reshape((len(train_numeric),1))))
    # train_numeric = np.hstack((train_numeric, np.array(train['T2_V1']/train['T1_V1']).reshape((len(train_numeric),1))))
    # train_numeric = np.hstack((train_numeric, np.array(train['T2_V1']/train['T1_V2']).reshape((len(train_numeric),1))))
    # train_numeric = np.hstack((train_numeric, np.array(train['T2_V1']*train['T2_V2']).reshape((len(train_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T1_V1']*test['T1_V2']).reshape((len(test_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T1_V1']*test['T1_V3']).reshape((len(test_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T1_V2']*test['T1_V3']).reshape((len(test_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T2_V1']/test['T1_V1']).reshape((len(test_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T2_V1']/test['T1_V2']).reshape((len(test_numeric),1))))
    # test_numeric = np.hstack((test_numeric, np.array(test['T2_V1']*test['T2_V2']).reshape((len(test_numeric),1))))



    feat_pack = (train_numeric, test_numeric, train_onehot, test_onehot, train_c2o, test_c2o, train_y)
    joblib.dump(feat_pack, './data/feat_pack_modified.pkl', compress = 3)

else:
    train_numeric, test_numeric, train_onehot, test_onehot, train_c2o, test_c2o, train_y = joblib.load('./data/feat_pack_modified.pkl')

##################################################################################
# change y

original_y = np.array(train_y)
indicator = 1/1.6
transformed_y = np.power(np.array(train_y), indicator)

##################################################################################
 # k-fold cross validation for random forest

kf = KFold(len(train_y), n_folds=3, shuffle=True, random_state=42)

# params = {}
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["max_depth"] = 7
# params["subsample"] = 0.65
# params["colsample_bytree"] = 0.4
# params["min_child_weight"] = 5
# params["silent"] = 1
# params["seed"] = 0
# plst = list(params.items())

# num_rounds = 100000

# print 'generating target ...'
# target = []
# for train_idx, test_idx in kf:
#     # the target
#     target = np.append(target, original_y[test_idx])

# output = []
# for train_idx, test_idx in kf:

#     train_data = train_numeric[train_idx]
#     test_data = train_numeric[test_idx]
#     train_target = transformed_y[train_idx]
#     test_target = transformed_y[test_idx]
#     eval_target = original_y[test_idx]

#     train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=42)

#     xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#     xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#     xgtest = xgb.DMatrix(test_data)

#     watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#     reg = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#     # xgtrain = xgb.DMatrix(train_data, label=train_target)
#     # xgtest = xgb.DMatrix(test_data)

#     # reg = xgb.train(plst, xgtrain, 900)
#     preds = reg.predict(xgtest)

#     output = np.append(output, preds)


reg = RandomForestRegressor(n_estimators=100, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)

output = [] # record cross validation scores

for train_idx, test_idx in kf:
    train_data = train_numeric[train_idx]
    test_data = train_numeric[test_idx]
    train_target = transformed_y[train_idx]
    eval_target = original_y[test_idx]

    reg.fit(train_data, train_target)
    preds = reg.predict(test_data)

    output = np.append(output, preds)



##################################################################################
# output
score = normalized_gini(target, output)

print 'CV score is: {}'.format(score)
print 'script duration is: {}'.format(datetime.now()-start_time)

run_log = '***********************************\n'
run_log += 'time is: {}\n'.format(datetime.now())
run_log += '{}\n'.format(reg)
run_log += '{}\n'.format(plst)
run_log += '{}\n'.format(kf)
run_log += 'y is transformed: power {}\n'.format(indicator)
run_log += 'CV score is: {}\n'.format(score)
run_log += 'script duration is: {}\n'.format(datetime.now()-start_time)

with open('./info/log.txt', 'a') as log_file:
    log_file.write(run_log)




















# ##################################################################################
# # change y

# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["max_depth"] = 7
# params["subsample"] = 0.65
# params["colsample_bytree"] = 0.4
# params["min_child_weight"] = 5
# params["silent"] = 1

# original_y = np.array(train_y)
# indicator = '2'

# y_1 = np.power(np.array(train_y), 2)

# transformed_ys = [y_1]

# for transformed_y in transformed_ys:

#     ##################################################################################
#     # k-fold cross validation for random forest

#     kf = KFold(len(train_y), n_folds=3, shuffle=True, random_state=42)

#     params = {}
#     params["objective"] = "reg:linear"
#     params["eta"] = 0.01
#     params["max_depth"] = 7
#     params["subsample"] = 0.8
#     params["colsample_bytree"] = 0.8
#     params["min_child_weight"] = 5
#     params["silent"] = 1
#     plst = list(params.items())
#     num_rounds = 100000

#     score = [] # record cross validation scores

#     for train_idx, test_idx in kf:

#         train_data = train_numeric[train_idx]
#         test_data = train_numeric[test_idx]
#         train_target = transformed_y[train_idx]
#         test_target = transformed_y[test_idx]
#         eval_target = original_y[test_idx]

#         train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=42)

#         xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
#         xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
#         xgtest = xgb.DMatrix(test_data)

#         watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#         reg = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

#         preds = reg.predict(xgtest)

#         score = np.append(score, normalized_gini(eval_target, preds))







# ##################################################################################
# # k-fold cross validation for linear models

# for lasso, alpha = 0.00035
# for ridge, alpha = 120

# kf = KFold(len(train_y), n_folds=3, shuffle=True, random_state=42)

# reg = Lasso(alpha=0.0001)

# score = [] # record cross validation scores

# for train_idx, test_idx in kf:
#     train_data = train_onehot[train_idx]
#     test_data = train_onehot[test_idx]
#     train_target = transformed_y[train_idx]
#     eval_target = original_y[test_idx]

#     reg.fit(train_data, train_target)
#     preds = reg.predict(test_data)

#     score = np.append(score, normalized_gini(eval_target, preds))




# ##################################################################################
# # k-fold cross validation for random forest

# use this:
# RandomForestRegressor(n_estimators=200, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)

# kf = KFold(len(train_y), n_folds=3, shuffle=True, random_state=42)

# reg = RandomForestRegressor(n_estimators=500, max_features=10, max_depth=15, min_samples_leaf=4, n_jobs=-1)

# score = [] # record cross validation scores

# for train_idx, test_idx in kf:
#     train_data = train_numeric[train_idx]
#     test_data = train_numeric[test_idx]
#     train_target = transformed_y[train_idx]
#     eval_target = original_y[test_idx]

#     reg.fit(train_data, train_target)
#     preds = reg.predict(test_data)

#     score = np.append(score, normalized_gini(eval_target, preds))


