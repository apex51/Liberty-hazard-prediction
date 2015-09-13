import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_extraction import DictVectorizer as DV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR
##################################################################################
# cal metric

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

##################################################################################
#transfer train and test and dunp

train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)
train_y = np.array(train.Hazard)
train_y = np.power(np.array(train.Hazard), 3)

# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)
# df_combine = pd.concat([train, test])

# columns and index for later use
columns = train.columns
test_ind = test.index

# df_combine.drop('T2_V10', axis=1, inplace=True)
# df_combine.drop('T2_V7', axis=1, inplace=True)
# df_combine.drop('T1_V13', axis=1, inplace=True)
# df_combine.drop('T1_V10', axis=1, inplace=True)

# drop noise, and linear model are sensitive to noise
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
# train.drop('T2_V11', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)
# test.drop('T2_V11', axis=1, inplace=True)



# columns and index for later use
columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

# train = train.astype(np.int64)
# test = test.astype(np.int64)

ohc = preprocessing.OneHotEncoder()
ohc.fit(np.vstack((train,test)))
train = ohc.transform(train)
test = ohc.transform(test)


# dict_combine = df_combine.T.to_dict().values()
# vectorizer = DV(sparse=False)
# vectorizer.fit(dict_combine)

# dict_train = train.T.to_dict().values()
# train = vectorizer.transform(dict_train).astype(np.float32)
# train = pd.DataFrame(train)
# dict_test = test.T.to_dict().values()
# test = vectorizer.transform(dict_test).astype(np.float32)
# test = pd.DataFrame(test)

# with open('./data/train_denoise.vec', 'wb') as f:
#     pickle.dump(train, f)

# with open('./data/test_denoise.vec', 'wb') as f:
#     pickle.dump(test, f)

# with open('./data/train_y.vec', 'wb') as f:
#     pickle.dump(train_y, f)

##################################################################################
#load train and test

# with open('./data/train_denoise.vec', 'rb') as f:
#     train = pickle.load(f)

# with open('./data/test_denoise.vec', 'rb') as f:
#     test = pickle.load(f)

# with open('./data/train_y.vec', 'rb') as f:
#     train_y = pickle.load(f)

##################################################################################
#tuning by cross validation

# for grid search score
# def gini_scorer(estimator, X, y):
#     pred = estimator.predict(X)
#     return normalized_gini(y, pred)

# modified scorer using ranks
def gini_scorer(estimator, X, y):
    pred = estimator.predict(X)
    pred = pred.argsort().argsort()
    return normalized_gini(y, pred)


param_grid = {'alpha': [0.001]}
grid_lasso = GridSearchCV(Lasso(), param_grid, gini_scorer)
grid_lasso.fit(train,train_y)

print 'Lasso\'s best para: {} with score {:.6f}'.format(grid_lasso.best_params_, grid_lasso.best_score_)

# param_grid = {'alpha': [100]}
# grid_rdg = GridSearchCV(Ridge(), param_grid, gini_scorer)
# grid_rdg.fit(train,train_y)
# print 'Ridge\'s best para: {} with score {:.6f}'.format(grid_rdg.best_params_, grid_rdg.best_score_)

# param_grid = {'alpha': [0.0015],
#               'l1_ratio': [0.97]}
# grid_el = GridSearchCV(ElasticNet(), param_grid, gini_scorer)
# grid_el.fit(train,train_y)
# print 'Ridge\'s best para: {} with score {:.6f}'.format(grid_el.best_params_, grid_el.best_score_)

# param_grid = {'dual': [True]}
# grid_svr = GridSearchCV(LinearSVR(), param_grid, gini_scorer)
# grid_svr.fit(train,train_y)
# print 'SVR\'s best para: {} with score {:.6f}'.format(grid_svr.best_params_, grid_svr.best_score_)

##################################################################################
#tuning by train test score

# train_x_sp, test_x_sp, train_y_sp, test_y_sp = train_test_split(train, train_y, train_size=0.8, random_state=50)

# rgrs = Lasso(alpha=0.01)
# rgrs.fit(train_x_sp, train_y_sp)
# pred = rgrs.predict(test_x_sp)

# score = normalized_gini(test_y_sp, pred)
# print '{:.6f}'.format(score)

##################################################################################
#train, predict and submit
# rgrs = Ridge(alpha=100)
# rgrs.fit(train,train_y)
# pred = rgrs.predict(test)

# rgrs = Lasso(alpha=0.001)
# rgrs.fit(train, train_y)
# pred = rgrs.predict(test)

# preds = pd.DataFrame({"Id": test_ind, "Hazard": pred})
# preds = preds.set_index('Id')
# preds.to_csv('lasso_output.csv')

# rgrs = ElasticNet(alpha=0.0015, l1_ratio=0.97)
# rgrs.fit(train,train_y)
# pred = rgrs.predict(test)

# preds = pd.DataFrame({"Id": test_ind, "Hazard": pred})
# preds = preds.set_index('Id')
# preds.to_csv('elasticnet_output.csv')

##################################################################################
# log
# for Ridge with noise
#  mean: 0.29264, std: 0.00815, params: {'normalize': True, 'alpha': 100},
#  mean: 0.33093, std: 0.00270, params: {'normalize': False, 'alpha': 100},--------
#  mean: 0.30336, std: 0.00684, params: {'normalize': True, 'alpha': 10},
#  mean: 0.33048, std: 0.00262, params: {'normalize': False, 'alpha': 10},
#  mean: 0.32578, std: 0.00417, params: {'normalize': True, 'alpha': 1},
#  mean: 0.33022, std: 0.00268, params: {'normalize': False, 'alpha': 1},
#  mean: 0.33066, std: 0.00295, params: {'normalize': True, 'alpha': 0.1},
#  mean: 0.33016, std: 0.00267, params: {'normalize': False, 'alpha': 0.1},
#  mean: 0.33027, std: 0.00272, params: {'normalize': True, 'alpha': 0.01},
#  mean: 0.33015, std: 0.00266, params: {'normalize': False, 'alpha': 0.01},
#  mean: 0.33019, std: 0.00269, params: {'normalize': True, 'alpha': 0.001},
#  mean: 0.33014, std: 0.00266, params: {'normalize': False, 'alpha': 0.001}
# for Lasso with noise
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 10},
#  mean: -0.00185, std: 0.00859, params: {'normalize': False, 'alpha': 10},
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 1},
#  mean: 0.14450, std: 0.00662, params: {'normalize': False, 'alpha': 1},
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 0.1},
#  mean: 0.25401, std: 0.01347, params: {'normalize': False, 'alpha': 0.1},-
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 0.01},
#  mean: 0.32503, std: 0.00394, params: {'normalize': False, 'alpha': 0.01},-
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 0.008},
#  mean: 0.32659, std: 0.00410, params: {'normalize': False, 'alpha': 0.008},-
#  mean: -0.00185, std: 0.00859, params: {'normalize': True, 'alpha': 0.006},
#  mean: 0.32799, std: 0.00434, params: {'normalize': False, 'alpha': 0.006}----------
# for Lasso and Ridge without noise
# Lasso's best para: {'normalize': False, 'alpha': 0.006} with score 0.328535
# Ridge's best para: {'normalize': False, 'alpha': 100} with score 0.331511
# for Lasso
#  mean: 0.32853, std: 0.00454, params: {'alpha': 0.006},
#  mean: 0.32919, std: 0.00463, params: {'alpha': 0.005},
#  mean: 0.32987, std: 0.00457, params: {'alpha': 0.004},
#  mean: 0.33062, std: 0.00439, params: {'alpha': 0.003},
#  mean: 0.33155, std: 0.00407, params: {'alpha': 0.002}
 # mean: 0.33062, std: 0.00439, params: {'alpha': 0.003},
 # mean: 0.33112, std: 0.00427, params: {'alpha': 0.0025},
 # mean: 0.33155, std: 0.00407, params: {'alpha': 0.002},
 # mean: 0.33179, std: 0.00377, params: {'alpha': 0.0015},
 # mean: 0.33177, std: 0.00345, params: {'alpha': 0.001}
 # mean: 0.33166, std: 0.00395, params: {'alpha': 0.0018},
 # mean: 0.33175, std: 0.00384, params: {'alpha': 0.0016},
 # mean: 0.33181, std: 0.00371, params: {'alpha': 0.0014},
 # mean: 0.33181, std: 0.00358, params: {'alpha': 0.0012}
 # 0.33183, std: 0.00365, params: {'alpha': 0.0013}------------------

# for elastic net
#  mean: 0.33136, std: 0.00284, params: {'alpha': 0.001, 'l1_ratio': 0.1},
#  mean: 0.33143, std: 0.00295, params: {'alpha': 0.001, 'l1_ratio': 0.3},
#  mean: 0.33149, std: 0.00309, params: {'alpha': 0.001, 'l1_ratio': 0.5},
#  mean: 0.33159, std: 0.00323, params: {'alpha': 0.001, 'l1_ratio': 0.7},
#  mean: 0.33152, std: 0.00293, params: {'alpha': 0.002, 'l1_ratio': 0.1},
#  mean: 0.33154, std: 0.00317, params: {'alpha': 0.002, 'l1_ratio': 0.3},
#  mean: 0.33158, std: 0.00347, params: {'alpha': 0.002, 'l1_ratio': 0.5},
#  mean: 0.33165, std: 0.00369, params: {'alpha': 0.002, 'l1_ratio': 0.7},-
#  mean: 0.33135, std: 0.00314, params: {'alpha': 0.005, 'l1_ratio': 0.1},
#  mean: 0.33105, std: 0.00362, params: {'alpha': 0.005, 'l1_ratio': 0.3},
#  mean: 0.33059, std: 0.00402, params: {'alpha': 0.005, 'l1_ratio': 0.5},
#  mean: 0.33007, std: 0.00435, params: {'alpha': 0.005, 'l1_ratio': 0.7},
#  mean: 0.33025, std: 0.00332, params: {'alpha': 0.01, 'l1_ratio': 0.1},
#  mean: 0.32922, std: 0.00385, params: {'alpha': 0.01, 'l1_ratio': 0.3},
#  mean: 0.32817, std: 0.00421, params: {'alpha': 0.01, 'l1_ratio': 0.5},
#  mean: 0.32708, std: 0.00433, params: {'alpha': 0.01, 'l1_ratio': 0.7}
# {'alpha': 0.0015, 'l1_ratio': 0.9} with score 0.331787
# {'alpha': 0.0015, 'l1_ratio': 0.97} with score 0.331792------------------


# for one hot drop noise
# 0.351972 for lasso 0.00135 
# [mean: 0.35219, std: 0.00479, params: {'alpha': 0.001}, -----
#  mean: 0.33384, std: 0.00659, params: {'alpha': 0.01}]
# [mean: 0.35216, std: 0.00466, params: {'alpha': 0.0005},
#  mean: 0.35086, std: 0.00519, params: {'alpha': 0.002},
#  mean: 0.34429, std: 0.00563, params: {'alpha': 0.005}]

# for ridge without noise
# [mean: 0.35313, std: 0.00433, params: {'alpha': 100}, -----
#  mean: 0.35157, std: 0.00436, params: {'alpha': 10}]
# [mean: 0.35308, std: 0.00441, params: {'alpha': 150},
#  mean: 0.35304, std: 0.00434, params: {'alpha': 80}]
# [mean: 0.34457, std: 0.00486, params: {'alpha': 1000},
#  mean: 0.35289, std: 0.00446, params: {'alpha': 200},
#  mean: 0.35157, std: 0.00436, params: {'alpha': 10},
#  mean: 0.35086, std: 0.00444, params: {'alpha': 1}]

# with noise
# Lasso's best para: {'alpha': 0.001} with score 0.350960
# Ridge's best para: {'alpha': 100} with score 0.351718
