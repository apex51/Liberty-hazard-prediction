'''
turn train_y into:
1's and 0's, train rf on it
1's, 2-10's, train xgb on it 

predict:
run rf on it, if 1 pass, if 0:
run xgb on it to get a num

submit label

'''

import pandas as pd
import numpy as np 
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

##################################################################################
def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True) # order submission
    rand = [float(i+1)/float(len(df)) for i in range(len(df))] # random diagonal curve
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
train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)
train_y = np.array(train.Hazard)

# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)

# columns and index for later use
columns = train.columns
test_ind = test.index

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

ohc = preprocessing.OneHotEncoder()
ohc.fit(np.vstack((train,test)))
train = ohc.transform(train)
test = ohc.transform(test)

# copy train_y
y_01 = np.array(train_y) # 0 stands for 1, 1 stands for others
y_110 = np.array(train_y) # num 1-10

for i in range(len(train_y)):
    if train_y[i] == 1:
        y_01[i] = 0
    else:
        y_01[i] = 1

for i in range(len(train_y)):
    if train_y[i] > 11:
        y_110[i] = 10
    if train_y[i] ==10 or train_y[i] ==11:
        y_110[i] = 9
    if train_y[i] ==9 or train_y[i] ==8:
        y_110[i] = 8

train_x_sp, test_x_sp, train_y_sp, test_y_sp, _, val_y = train_test_split(train, y_01, train_y, train_size=0.8, random_state=50)

clf = LogisticRegression(penalty='l1')
clf.fit(train_x_sp, train_y_sp)
pred = clf.predict(test_x_sp)

print accuracy_score(test_y_sp,pred)
print normalized_gini(val_y, pred)



# train_idx = (train_y_sp != 1)
# test_idx = (test_y_sp != 1)

# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features=10)
# clf.fit(train_x_sp, train_y_sp)
# pred = clf.predict(test_x_sp)

# print accuracy_score(test_y_sp,pred)
# print normalized_gini(val_y, pred)


# params = {}
# params["objective"] = "reg:logistic"
# params["eval_metric"] = "logloss"
# params["eta"] = 0.01
# params["max_depth"] = 7
# params["subsample"] = 0.8
# params["colsample_bytree"] = 0.8
# params["min_child_weight"] = 5
# params["silent"] = 1

# plst = list(params.items())

# num_rounds = 10000

# #create a train and validation dmatrices 
# xgtrain = xgb.DMatrix(train_x_sp, label=train_y_sp)
# xgval = xgb.DMatrix(test_x_sp, label=test_y_sp)

# watchlist = [(xgtrain, 'train'),(xgval, 'val')]
# rgrs = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30)
# pred = rgrs.predict(xgval)

# print 'gini score is {}'.format(normalized_gini(val_y, pred))
# print accuracy_score(test_y_sp,pred)
# # class_reg.accuracy_score(class_reg.test_y_sp,pred)
# # Out[21]: 0.65960784313725496

























