import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA

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

# gini wrapper for xgboost
# return 0 - gini score for decrease purpose
def gini_xgb(pred, dtrain):
    target = dtrain.get_label()
    err = 0. - normalized_gini(target, pred)
    return 'gini', err

##################################################################################
#load train and test 

train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)
train_y = np.power(np.array(train.Hazard), 0.5)

# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)
# # drop noisy features
train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
# train.drop('T1_V6', axis=1, inplace=True)

# train.drop('T2_V11', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)
# test.drop('T1_V6', axis=1, inplace=True)

# test.drop('T2_V11', axis=1, inplace=True)

# columns and index for later use
columns = train.columns
test_ind = test.index

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

train = train.astype(np.int64)
test = test.astype(np.int64)

# pca did not perform in boost
# pca = PCA()
# pca.fit(np.vstack((train, test)))
# train = pca.transform(train)
# test = pca.transform(test)

# ohc did not perform in boost
# ohc = preprocessing.OneHotEncoder()
# ohc.fit(np.vstack((train,test)))
# train = ohc.transform(train)
# test = ohc.transform(test)



##################################################################################

# with open('./data/train_denoise.vec', 'rb') as f:
#     train = pickle.load(f)

# with open('./data/test_denoise.vec', 'rb') as f:
#     test = pickle.load(f)

# with open('./data/train_y.vec', 'rb') as f:
#     train_y = pickle.load(f)

train_x_sp, test_x_sp, train_y_sp, test_y_sp = train_test_split(train, train_y, train_size=0.8, random_state=50)

params = {}
params["objective"] = "count:poisson"
params["eta"] = 0.01
params["max_depth"] = 7
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["min_child_weight"] = 5
params["silent"] = 1

plst = list(params.items())

num_rounds = 100000

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train_x_sp, label=train_y_sp)
xgval = xgb.DMatrix(test_x_sp, label=test_y_sp)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
rgrs = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

print 'gini score is {}'.format(normalized_gini(test_y_sp, rgrs.predict(xgval)))

# cat to numeric
# with noise, 0.374878846709
# without noise, 0.378314390403 (719 rounds) 0.377940652216(564) 0.376572609052(T1_V6 seems useful)
# the origin is with noise, and the vec one is without noise, parameters are the same
# cat to onehot: gini score is 0.376673478783(with noise)
# cat to onehot: gini score is 0.37717226583 (with noise)

##################################################################################
# bst linear doesn't work

# train_x_sp, test_x_sp, train_y_sp, test_y_sp = train_test_split(train, train_y, train_size=0.8, random_state=50)

# params = {}
# params["task"] = "regression"
# params["booster"] = "gblinear"
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["alpha"] = 0.0002
# params["lambda"] = 200
# params["lambda_bias"] = 10
# params["silent"] = 1


# plst = list(params.items())

# num_rounds = 10000

# #create a train and validation dmatrices 
# xgtrain = xgb.DMatrix(train_x_sp, label=train_y_sp)
# xgval = xgb.DMatrix(test_x_sp, label=test_y_sp)

# watchlist = [(xgtrain, 'train'),(xgval, 'val')]
# rgrs = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)

# print 'gini score is {}'.format(normalized_gini(test_y_sp, rgrs.predict(xgval)))



##################################################################################
# boost linear, log
# raw feat, 0.01: 0.267863761405
# raw feat, lambda 2: 0.267866393147
# eta 0.01, alpha 0.1, 0.31171946783
# eta 0.01, aplha 0.01, 0.311719930441
# eta 0.01, alpha 0.001, 0.311727609781
# lambda 2, 0.31182835615
# bias 1, 0.31182112914
# lambda 100, 0.313257422829
# for one hot
# undrop, gini score is 0.344490822327 --------
# undrop, 0.345063236465

##################################################################################
# boost tree, train, predict and submit

# params = {}
# params["objective"] = "reg:linear"
# params["eta"] = 0.01
# params["max_depth"] = 7
# params["subsample"] = 0.8
# params["colsample_bytree"] = 0.8
# params["min_child_weight"] = 5
# params["silent"] = 1

# plst = list(params.items())

# xgtrain = xgb.DMatrix(train, label=train_y)
# xgtest = xgb.DMatrix(test)

# num_rounds = 600

# watchlist = [(xgtrain, 'train')]
# rgrs = xgb.train(plst, xgtrain, num_rounds, watchlist)
# pred = rgrs.predict(xgtest)

# #generate solution
# preds = pd.DataFrame({"Id": test_ind, "Hazard": pred})
# preds = preds.set_index('Id')
# preds.to_csv('xgb_0.5.csv')

##################################################################################

# def xgboost_pred(train,labels,test):
#     params = {}
#     params["objective"] = "reg:linear"
#     params["eta"] = 0.005
#     params["min_child_weight"] = 6
#     params["subsample"] = 0.7
#     params["colsample_bytree"] = 0.7
#     params["scale_pos_weight"] = 1
#     params["silent"] = 1
#     params["max_depth"] = 9
    
    
#     plst = list(params.items())

#     #Using 5000 rows for early stopping. 
#     offset = 4000

#     num_rounds = 10000
#     xgtest = xgb.DMatrix(test)

#     #create a train and validation dmatrices 
#     xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
#     xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#     #train using early stopping and predict
#     watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#     model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
#     preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


#     #reverse train and labels and use different 5k for early stopping. 
#     # this adds very little to the score but it is an option if you are concerned about using all the data. 
#     train = train[::-1,:]
#     labels = np.log(labels[::-1])

#     xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
#     xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#     watchlist = [(xgtrain, 'train'),(xgval, 'val')]
#     model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
#     preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


#     #combine predictions
#     #since the metric only cares about relative rank we don't need to average
#     preds = (preds1)*1.4 + (preds2)*8.6
#     return preds

##################################################################################
# for xgb tree tuning
# default 0.01, 7, 0.8, .08, 5.6
# eta 0.005, 0.375781554733
# eta 0.01, depth 7, 0.376150872416, max_iter 605
# depth 5, 0.373860701851
# depth 10, 0.369762051727
# depth 8, 0.372068423952
# depth 6, 0.375179626021
# child_weight 1, 0.3752408243
# child_weight 3, 0.3749674007
# child_weight 5, 0.376183357
# child_weight 4, 0.3753828561
# child_weight 6, 0.376150872416
# subsample 0.5, 0.375326211987
# subsample 0.7, 0.37412052468
# subsample 0.9, 0.37437878491
# eta 0.001, 0.37542800694
# subsample 0.5, 0.373855757055
# subsample 0.7, 0.37519109877
# col_sub_sample 0.6, 0.375161697279
# eta 0.05, 0.371400290484
# eta 0.02, 0.375141846132
# eta 0.008, 0.375257272686
# eta 0.01, 0.37618335798
# eta 0.009, 0.375051061318

# for pca
# 0.698169619615
# 0.768678243934
# 0.825113259336
# 0.870863662254
# 0.9035033877
# 0.931808852793
# 0.957723481516
# 0.9660868074
# 0.973973578659
# 0.980437187332
# 0.98532831741
# 0.989051809886
# 0.991998076633
# 0.994300451271
# 0.995945889982
# 0.996830423195
# 0.997492974615
# 0.997891812281
# 0.998266712391
# 0.998577242992
# 0.99888108692
# 0.999155217403
# 0.999407996275
# 0.999634686073
# 0.999844214312
# 0.999951585073
# 0.999993741111
# 1.0

# for one hot code:
#     gini score is 0.354245990404
