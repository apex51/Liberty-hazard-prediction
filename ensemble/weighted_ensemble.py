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
# train and predict 
##################################################################################

# record predictions and target

print 'generating rf ...'
train_data = train_numeric
train_target = transformed_y
test_data = test_numeric
# the target
reg_1 = RandomForestRegressor(n_estimators=1000, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
reg_1.fit(train_data, train_target)
pred_1 = reg_1.predict(test_data)
rf = np.array(pred_1)

print 'generating rf order...'
train_data = train_numeric_order
train_target = transformed_y
test_data = test_numeric_order
# the target
reg_11 = RandomForestRegressor(n_estimators=1000, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
reg_11.fit(train_data, train_target)
pred_11 = reg_11.predict(test_data)
rf_order = np.array(pred_11)

print 'generating rf relate...'
train_data = train_relate
train_target = transformed_y
test_data = test_relate
# the target
reg_12 = RandomForestRegressor(n_estimators=1000, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
reg_12.fit(train_data, train_target)
pred_12 = reg_12.predict(test_data)
rf_relate = np.array(pred_12)

print 'generating rf with log_y...'
train_data = train_numeric
train_target = log_y
test_data = test_numeric
# the target
reg_13 = RandomForestRegressor(n_estimators=1000, max_features=8, max_depth=14, min_samples_leaf=4, n_jobs=-1)
reg_13.fit(train_data, train_target)
pred_13 = reg_13.predict(test_data)
rf_log_y = np.array(pred_13)


# para for xgb
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["max_depth"] = 7
params["subsample"] = 0.65
params["colsample_bytree"] = 0.4
params["min_child_weight"] = 5
params["silent"] = 0
# plst = list(params.items())
num_rounds = 100000

print 'generating xgb on cat to onehot ...'

xgb_c2o = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_c2o
    test_data = test_c2o
    train_target = transformed_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_3 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_3 = reg_3.predict(xgtest, ntree_limit=reg_3.best_iteration)

    xgb_1round = np.array(pred_3)

    xgb_c2o += xgb_1round

xgb_c2o = xgb_c2o/5

print 'generating xgb on relate ...'

xgb_relate = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_relate
    test_data = test_relate
    train_target = transformed_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_31 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_31 = reg_31.predict(xgtest, ntree_limit=reg_31.best_iteration)

    xgb_1round = np.array(pred_31)

    xgb_relate += xgb_1round

xgb_relate = xgb_relate/5


print 'generating xgb on log no order ...'

xgb_log_no_order = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_log
    test_data = test_log
    train_target = transformed_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_32 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_32 = reg_32.predict(xgtest, ntree_limit=reg_32.best_iteration)

    xgb_1round = np.array(pred_32)

    xgb_log_no_order += xgb_1round

xgb_log_no_order = xgb_log_no_order/5

print 'generating xgb on log with order ...'

xgb_log_order = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_log_order
    test_data = test_log_order
    train_target = transformed_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_33 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_33 = reg_33.predict(xgtest, ntree_limit=reg_33.best_iteration)

    xgb_1round = np.array(pred_33)

    xgb_log_order += xgb_1round

xgb_log_order = xgb_log_order/5


print 'generating xgb on numeric with log_y ...'

xgb_numeric_log_y = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_numeric
    test_data = test_numeric
    train_target = log_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_34 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_34 = reg_34.predict(xgtest, ntree_limit=reg_34.best_iteration)

    xgb_1round = np.array(pred_34)

    xgb_numeric_log_y += xgb_1round

xgb_numeric_log_y = xgb_numeric_log_y/5

print 'generating xgb on cat to onehot with log_y...'

xgb_c2o_log_y = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_c2o
    test_data = test_c2o
    train_target = log_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_35 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_35 = reg_35.predict(xgtest, ntree_limit=reg_35.best_iteration)

    xgb_1round = np.array(pred_35)

    xgb_c2o_log_y += xgb_1round

xgb_c2o_log_y = xgb_c2o_log_y/5

print 'generating xgb on log with order with log_y ...'

xgb_log_order_log_y = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_log_order
    test_data = test_log_order
    train_target = log_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_36 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_36 = reg_36.predict(xgtest, ntree_limit=reg_36.best_iteration)

    xgb_1round = np.array(pred_36)

    xgb_log_order_log_y += xgb_1round

xgb_log_order_log_y = xgb_log_order_log_y/5

print 'generating xgb on log no order with log_y...'

xgb_log_no_order_log_y = np.zeros(len(test_data))

for i in range(5):

    params["seed"] = i
    plst = list(params.items())

    train_data = train_log
    test_data = test_log
    train_target = log_y

    train_sp_idx, eval_sp_idx = train_test_split(range(len(train_data)), train_size=0.9, random_state=i)
        
    xgtrain = xgb.DMatrix(train_data[train_sp_idx], label=train_target[train_sp_idx])
    xgval = xgb.DMatrix(train_data[eval_sp_idx], label=train_target[eval_sp_idx])
    xgtest = xgb.DMatrix(test_data)

    watchlist = [(xgval, 'val')]
    reg_32 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

    pred_32 = reg_32.predict(xgtest, ntree_limit=reg_32.best_iteration)

    xgb_1round = np.array(pred_32)

    xgb_log_no_order_log_y += xgb_1round

xgb_log_no_order_log_y = xgb_log_no_order_log_y/5



# print 'generating lasso on onehot ...'
# lasso_onehot = []
# for train_idx, test_idx in kf:
#     train_data = train_onehot[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_onehot[test_idx]
#     # the target
#     reg_4 = Lasso(alpha=0.00035)
#     reg_4.fit(train_data, train_target)
#     pred_4 = reg_4.predict(test_data)
#     lasso_onehot = np.append(lasso_onehot, pred_4)

# print 'generating ridge on onehot ...'
# ridge_onehot = []
# for train_idx, test_idx in kf:
#     train_data = train_onehot[train_idx]
#     train_target = transformed_y[train_idx]
#     test_data = train_onehot[test_idx]
#     # the target
#     reg_5 = Ridge(alpha=120)
#     reg_5.fit(train_data, train_target)
#     pred_5 = reg_5.predict(test_data)
#     ridge_onehot = np.append(ridge_onehot, pred_5)


##################################################################################
# use colyba to min func
##################################################################################
# rf, xgb_c2o, xgb_relate, rf_order, rf_relate, xgb_log_no_order, xgb_log_order

# result = pd.DataFrame()
# result['rf'] = rf
# result['rf_order'] = rf_order
# result['rf_relate'] = rf_relate
# result['xgb_relate'] = xgb_relate
# result['xgb_c2o'] = xgb_c2o
# result['xgb_log'] = xgb_log
# result['xgb_log_order'] = xgb_log_order
# result.to_csv('./info/ensemble_result.csv')


predictions = hstack((rf.reshape(len(rf),1),
                      xgb_c2o.reshape(len(xgb_c2o),1),
                      xgb_relate.reshape(len(xgb_relate),1),
                      rf_order.reshape(len(rf_order),1),
                      rf_relate.reshape(len(rf_relate),1),
                      xgb_log_no_order.reshape(len(xgb_log_no_order),1),
                      xgb_log_order.reshape(len(xgb_log_order),1),
                      rf_log_y.reshape(len(rf_log_y),1),
                      xgb_numeric_log_y.reshape(len(xgb_numeric_log_y),1),
                      xgb_c2o_log_y.reshape(len(xgb_c2o_log_y),1),
                      xgb_log_order_log_y.reshape(len(xgb_log_order_log_y),1),
                      xgb_log_no_order_log_y.reshape(len(xgb_log_no_order_log_y),1)))


w0 = array([4.67580555e-01, 2.11711896e+00, 1.70813157e+00,
            6.05814076e-05, 5.69982450e-01, 1.79998417e+00,
            1.16805668e+00, 2.64736293e-02, 7.27634375e-01,
            2.60181138e+00, 1.56868769e-04, 1.66585218e+00])

preds = np.dot(predictions, w0)

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('output_final_ensemble_final.csv')

