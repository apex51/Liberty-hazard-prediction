import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import pickle
from sklearn.decomposition import PCA

##################################################################################
# cal metric

# def gini(solution, submission):
#     df = zip(solution, submission, range(len(solution)))
#     df = sorted(df, key=lambda x: x[1], reverse=True) # order submission
#     rand = [float(i+1)/float(len(df)) for i in range(len(df))] # random diagonal curve
#     totalPos = float(sum([x[0] for x in df]))
#     cumPosFound = [df[0][0]]
#     for i in range(1,len(df)):
#         cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
#     Lorentz = [float(x)/totalPos for x in cumPosFound]
#     Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
#     return sum(Gini)

# def normalized_gini(solution, submission):
#     # solution = solution.argsort().argsort()
#     # submission = submission.argsort().argsort()
#     normalized_gini = gini(solution, submission)/gini(solution, solution)
#     return normalized_gini

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
#load train and test 

train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)
origin_y = np.array(train.Hazard)
train_y = np.power(np.array(train.Hazard), 0.5)
# train_y = np.log(np.array(train.Hazard))


# drop train_y -> train_y
train.drop('Hazard', axis=1, inplace=True)
# drop noisy features
# train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
train.drop('T1_V13', axis=1, inplace=True)
train.drop('T1_V10', axis=1, inplace=True)
train.drop('T1_V6', axis=1, inplace=True)
# train.drop('T2_V11', axis=1, inplace=True)

# test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
test.drop('T1_V13', axis=1, inplace=True)
test.drop('T1_V10', axis=1, inplace=True)
test.drop('T1_V6', axis=1, inplace=True)

# columns and index for later use
columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# for i in range(train.shape[1]):
#     if type(train[1,i]) is str:
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(train[:,i]))
#         train[:,i] = lbl.transform(train[:,i])


# label encode the categorical variables
for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(int)
test = test.astype(int)

# train = np.column_stack((train, np.multiply(train[:,0], train[:,3])))
# train = np.column_stack((train, np.multiply(train[:,0], train[:,7])))
# train = np.column_stack((train, np.multiply(train[:,0], train[:,14])))
# train = np.column_stack((train, np.multiply(train[:,0], train[:,24])))
# train = np.column_stack((train, np.multiply(train[:,2], train[:,24])))
# train = np.column_stack((train, np.multiply(train[:,9], train[:,12])))
# train = np.column_stack((train, np.multiply(train[:,9], train[:,23])))
# train = np.column_stack((train, np.multiply(train[:,20], train[:,21])))
# train = np.column_stack((train, np.multiply(train[:,21], train[:,25])))
# train = np.column_stack((train, np.multiply(train[:,23], train[:,24])))



# dict_combine = df_combine.T.to_dict().values()
# vectorizer = DV(sparse=False)
# vectorizer.fit(dict_combine)

# dict_train = train.T.to_dict().values()
# train = vectorizer.transform(dict_train).astype(np.float32)
# train = pd.DataFrame(train)
# dict_test = test.T.to_dict().values()
# test = vectorizer.transform(dict_test).astype(np.float32)
# test = pd.DataFrame(test)





# ohc seems do not help
# ohc = preprocessing.OneHotEncoder()
# ohc.fit(np.vstack((train,test)))
# train = ohc.fit_transform(train)
# test = ohc.transform(test)

# # sparse to array
# train = train.toarray()
# test = test.toarray()

# # pca seems do not help
# pca = PCA(275)
# pca.fit(np.vstack((train, test)))
# train = pca.transform(train)
# test = pca.transform(test)

# 274, 0.98988561906
# 275, 0.99028894519


##################################################################################
# with open('./data/train_denoise.vec', 'rb') as f:
#     train = pickle.load(f)

# with open('./data/test_denoise.vec', 'rb') as f:
#     test = pickle.load(f)

# with open('./data/train_y.vec', 'rb') as f:
#     train_y = pickle.load(f)

# grid search score
def gini_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return normalized_gini(y, pred)

# param_grid = {'max_features': [10],
#               'max_depth': [15],
#               'min_samples_leaf': [4]}
# grid_rf = GridSearchCV(RandomForestRegressor(n_estimators=100, n_jobs=-1), param_grid, gini_scorer)
# grid_rf.fit(train,train_y)
# print 'RF\'s best para: {} with score {:.6f}'.format(grid_rf.best_params_, grid_rf.best_score_)

# param_grid = {'max_features': [15],
#               'max_depth': [15],
#               'min_samples_leaf': [4]}
# grid_et = RandomizedSearchCV(ExtraTreesRegressor(n_estimators=500, n_jobs=-1), param_grid, 1, gini_scorer)
# grid_et.fit(train,train_y)
# print 'ET\'s best para: {} with score {:.6f}'.format(grid_et.best_params_, grid_et.best_score_)

train_x_sp, test_x_sp, train_y_sp, test_y_sp, _, eval_y_sp = train_test_split(train, train_y, origin_y, train_size=0.8, random_state=50)

# rgrs = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rgrs = RandomForestRegressor(n_estimators=100, max_features=10, max_depth=15, min_samples_leaf=4, n_jobs=-1)
# rgrs = ExtraTreesRegressor(n_estimators=500, max_features=15, max_depth=15, min_samples_leaf=4, n_jobs=-1)
rgrs.fit(train_x_sp, train_y_sp)
# rf = RandomForestRegressor(n_estimators=400, n_jobs=-1)
# et = ExtraTreesRegressor(n_estimators=400, n_jobs=-1)

# rgrs.fit(train, train_y)
# et.fit(train, train_y)

pred = rgrs.predict(test_x_sp)

score = normalized_gini(eval_y_sp.argsort().argsort(), pred.argsort().argsort())
print '{:.6f}'.format(score)
score = normalized_gini(test_y_sp.argsort().argsort(), pred.argsort().argsort())
print '{:.6f}'.format(score)



##################################################################################
# parameter for RF is just max_features 10 and max_depth 15
# with min_leaf_size 4, 0.365329

# parameter for ET is just max_features 15 and max_depth 15
# with min_leaf_size 4, 0.0.356783

# cat to onehot: 0.345079 (seems rf likes numeric numbers, so give up cat->onehot)

##################################################################################
# log of nm-gini:  
# rf 100: 0.324915, 0.321849
# rf 500: 0.331215, 0.330033
# rf 500 max_feature 15: 0.339017
# rf 500 max_feature 6: 0.347608
# rf 500 max_feature 5: 0.348715
# rf 500 max_feature 4: 0.349051
# rf 500 max_feature 5 max_depth 15: 0.354515
# for RF
#  mean: 0.30875, std: 0.00743, params: {'max_features': 4, 'max_depth': 5},
#  mean: 0.35105, std: 0.00580, params: {'max_features': 15, 'max_depth': 20},
#  mean: 0.35549, std: 0.00747, params: {'max_features': 6, 'max_depth': None},
#  mean: 0.34693, std: 0.00683, params: {'max_features': 15, 'max_depth': 50},
#  mean: 0.35970, std: 0.00661, params: {'max_features': 6, 'max_depth': 20},
#  mean: 0.35748, std: 0.00994, params: {'max_features': 3, 'max_depth': 50},
#  mean: 0.35780, std: 0.00966, params: {'max_features': 4, 'max_depth': 50},
#  mean: 0.35989, std: 0.00812, params: {'max_features': 3, 'max_depth': 15},
#  mean: 0.35818, std: 0.00934, params: {'max_features': 4, 'max_depth': None},
#  mean: 0.34035, std: 0.00735, params: {'max_features': 25, 'max_depth': None},
#--mean: 0.36192, std: 0.00644, params: {'max_features': 10, 'max_depth': 15}-------
#  with min_leaf 2, 0.360062
#  mean: 0.36185, std: 0.00802, params: {'max_features': 4, 'max_depth': 15},
#  mean: 0.31292, std: 0.00561, params: {'max_features': 20, 'max_depth': 5},
#  mean: 0.35648, std: 0.00698, params: {'max_features': 10, 'max_depth': 20},
#  mean: 0.35443, std: 0.01026, params: {'max_features': 2, 'max_depth': None},
#  mean: 0.34597, std: 0.00897, params: {'max_features': 3, 'max_depth': 10},
#  mean: 0.35280, std: 0.00771, params: {'max_features': 5, 'max_depth': 10},
#  mean: 0.34045, std: 0.00803, params: {'max_features': 25, 'max_depth': 30},
#  mean: 0.34391, std: 0.00783, params: {'max_features': 20, 'max_depth': None},
#  mean: 0.35729, std: 0.00653, params: {'max_features': 10, 'max_depth': 10},
#  mean: 0.34426, std: 0.00672, params: {'max_features': 20, 'max_depth': 50},
#  mean: 0.35651, std: 0.00750, params: {'max_features': 5, 'max_depth': 30},
#  mean: 0.31294, std: 0.00668, params: {'max_features': 10, 'max_depth': 5},
#  mean: 0.35016, std: 0.00811, params: {'max_features': 4, 'max_depth': 10},
#  mean: 0.35728, std: 0.00535, params: {'max_features': 15, 'max_depth': 10},
#  mean: 0.35135, std: 0.00577, params: {'max_features': 10, 'max_depth': 50},
#  mean: 0.36180, std: 0.00830, params: {'max_features': 4, 'max_depth': 20},
#  mean: 0.35729, std: 0.00953, params: {'max_features': 3, 'max_depth': None},
#  mean: 0.34741, std: 0.00781, params: {'max_features': 15, 'max_depth': 30},
#  mean: 0.35661, std: 0.00979, params: {'max_features': 3, 'max_depth': 30},
#  mean: 0.35322, std: 0.01026, params: {'max_features': 2, 'max_depth': 50},
#  mean: 0.35145, std: 0.00739, params: {'max_features': 10, 'max_depth': None},
#  mean: 0.35564, std: 0.00578, params: {'max_features': 20, 'max_depth': 10},
#  mean: 0.35913, std: 0.00532, params: {'max_features': 15, 'max_depth': 15},
#  mean: 0.30618, std: 0.00819, params: {'max_features': 3, 'max_depth': 5},
#  mean: 0.36030, std: 0.00926, params: {'max_features': 3, 'max_depth': 20},
#  mean: 0.34376, std: 0.00672, params: {'max_features': 20, 'max_depth': 30},
#  mean: 0.35871, std: 0.00812, params: {'max_features': 5, 'max_depth': 50},
#  mean: 0.34567, std: 0.00788, params: {'max_features': 25, 'max_depth': 20},
#  mean: 0.36015, std: 0.00787, params: {'max_features': 5, 'max_depth': 20}

# for ET
#  mean: 0.34238, std: 0.00900, params: {'max_features': 4, 'max_depth': None},
#  mean: 0.29548, std: 0.00717, params: {'max_features': 4, 'max_depth': 5},
#  mean: 0.33226, std: 0.00911, params: {'max_features': 1, 'max_depth': 20},
#  mean: 0.34270, std: 0.00868, params: {'max_features': 4, 'max_depth': 30},
#  mean: 0.33708, std: 0.00710, params: {'max_features': 20, 'max_depth': 50},
#  mean: 0.35362, std: 0.00569, params: {'max_features': 20, 'max_depth': 15},
#  mean: 0.35310, std: 0.00510, params: {'max_features': 25, 'max_depth': 15},
#--mean: 0.35419, std: 0.00730, params: {'max_features': 15, 'max_depth': 15}------
# {'max_features': 15, 'max_depth': 15, 'min_samples_leaf': 2} with score 0.356744
#  mean: 0.33458, std: 0.00674, params: {'max_features': 25, 'max_depth': 50},
#  mean: 0.34733, std: 0.00973, params: {'max_features': 3, 'max_depth': 20},
#  mean: 0.33539, std: 0.00705, params: {'max_features': 25, 'max_depth': 30},
#  mean: 0.32683, std: 0.01060, params: {'max_features': 1, 'max_depth': 50},
#  mean: 0.34331, std: 0.00795, params: {'max_features': 10, 'max_depth': 30},
#  mean: 0.34475, std: 0.00720, params: {'max_features': 6, 'max_depth': 50},
#  mean: 0.33655, std: 0.00742, params: {'max_features': 6, 'max_depth': 10},
#  mean: 0.35103, std: 0.00945, params: {'max_features': 6, 'max_depth': 20},
#  mean: 0.31704, std: 0.00950, params: {'max_features': 2, 'max_depth': 10},
#  mean: 0.34306, std: 0.00782, params: {'max_features': 10, 'max_depth': None},
#  mean: 0.34176, std: 0.01060, params: {'max_features': 3, 'max_depth': None},
#  mean: 0.34374, std: 0.00946, params: {'max_features': 3, 'max_depth': 30},
#  mean: 0.34247, std: 0.00848, params: {'max_features': 3, 'max_depth': 15},
#  mean: 0.35008, std: 0.00895, params: {'max_features': 4, 'max_depth': 20},
#  mean: 0.33293, std: 0.00848, params: {'max_features': 5, 'max_depth': 10},
#  mean: 0.29363, std: 0.00695, params: {'max_features': 3, 'max_depth': 5},
#  mean: 0.35123, std: 0.00777, params: {'max_features': 5, 'max_depth': 20},
#  mean: 0.35010, std: 0.00877, params: {'max_features': 10, 'max_depth': 20},
#  mean: 0.34174, std: 0.00688, params: {'max_features': 25, 'max_depth': 20},
#  mean: 0.33690, std: 0.00671, params: {'max_features': 20, 'max_depth': 30},
#  mean: 0.34291, std: 0.00882, params: {'max_features': 10, 'max_depth': 50},
#  mean: 0.28897, std: 0.00591, params: {'max_features': 1, 'max_depth': 5},
#  mean: 0.35202, std: 0.00863, params: {'max_features': 6, 'max_depth': 15},
#  mean: 0.34348, std: 0.00868, params: {'max_features': 5, 'max_depth': None},
#  mean: 0.29220, std: 0.00773, params: {'max_features': 2, 'max_depth': 5},
#  mean: 0.34209, std: 0.00804, params: {'max_features': 10, 'max_depth': 10},
#  mean: 0.34103, std: 0.00727, params: {'max_features': 15, 'max_depth': 30},
#  mean: 0.33728, std: 0.01172, params: {'max_features': 2, 'max_depth': 50},
#  mean: 0.30580, std: 0.00609, params: {'max_features': 25, 'max_depth': 5},
#  mean: 0.33995, std: 0.00839, params: {'max_features': 15, 'max_depth': None},
#  mean: 0.30185, std: 0.00622, params: {'max_features': 10, 'max_depth': 5},
#  mean: 0.34473, std: 0.00781, params: {'max_features': 6, 'max_depth': 30},
#  mean: 0.34756, std: 0.00574, params: {'max_features': 25, 'max_depth': 10},
#  mean: 0.30833, std: 0.00877, params: {'max_features': 1, 'max_depth': 10},
#  mean: 0.34673, std: 0.00903, params: {'max_features': 4, 'max_depth': 15},
#  mean: 0.34060, std: 0.00832, params: {'max_features': 3, 'max_depth': 50},
#  mean: 0.33400, std: 0.00831, params: {'max_features': 2, 'max_depth': 15},
#  mean: 0.32846, std: 0.00984, params: {'max_features': 1, 'max_depth': None},
#  mean: 0.34465, std: 0.01011, params: {'max_features': 6, 'max_depth': None},
#  mean: 0.34686, std: 0.00607, params: {'max_features': 20, 'max_depth': 10},
#  mean: 0.32388, std: 0.00769, params: {'max_features': 3, 'max_depth': 10},
#  mean: 0.33804, std: 0.00759, params: {'max_features': 20, 'max_depth': None}

# for RF using min_samples_split
# [mean: 0.35115, std: 0.00561, params: {'max_features': 15, 'min_samples_split': 7, 'max_depth': 17},
#  mean: 0.35298, std: 0.00589, params: {'max_features': 15, 'min_samples_split': 9, 'max_depth': 17},
#  mean: 0.35496, std: 0.00808, params: {'max_features': 10, 'min_samples_split': 3, 'max_depth': 17},
#  mean: 0.35675, std: 0.00656, params: {'max_features': 10, 'min_samples_split': 9, 'max_depth': 17},
#  mean: 0.35551, std: 0.00739, params: {'max_features': 10, 'min_samples_split': 5, 'max_depth': 17},
#  mean: 0.35440, std: 0.00770, params: {'max_features': 10, 'min_samples_split': 9, 'max_depth': 19},
#  mean: 0.35091, std: 0.00666, params: {'max_features': 15, 'min_samples_split': 3, 'max_depth': 17},
#  mean: 0.35203, std: 0.00675, params: {'max_features': 15, 'min_samples_split': 5, 'max_depth': 17},
#  mean: 0.35291, std: 0.00764, params: {'max_features': 10, 'min_samples_split': 3, 'max_depth': 19},
#  mean: 0.35811, std: 0.00738, params: {'max_features': 10, 'min_samples_split': 5, 'max_depth': 15},
#  mean: 0.34783, std: 0.00630, params: {'max_features': 15, 'min_samples_split': 5, 'max_depth': 19},
#  mean: 0.35523, std: 0.00742, params: {'max_features': 10, 'min_samples_split': 7, 'max_depth': 17},
#  mean: 0.35522, std: 0.00626, params: {'max_features': 15, 'min_samples_split': 3, 'max_depth': 15},
#  mean: 0.35457, std: 0.00654, params: {'max_features': 15, 'min_samples_split': 7, 'max_depth': 15},
#  mean: 0.35732, std: 0.00695, params: {'max_features': 10, 'min_samples_split': 3, 'max_depth': 15},
#  mean: 0.35468, std: 0.00687, params: {'max_features': 15, 'min_samples_split': 5, 'max_depth': 15},
#  mean: 0.34747, std: 0.00650, params: {'max_features': 15, 'min_samples_split': 3, 'max_depth': 19},
#  mean: 0.34794, std: 0.00664, params: {'max_features': 15, 'min_samples_split': 7, 'max_depth': 19},
#  mean: 0.35305, std: 0.00777, params: {'max_features': 10, 'min_samples_split': 5, 'max_depth': 19},
#  mean: 0.35469, std: 0.00776, params: {'max_features': 10, 'min_samples_split': 7, 'max_depth': 19}]

##################################################################################
# train, predict and submit

# rgrs = RandomForestRegressor(n_estimators=500, max_features=10, max_depth=15, min_samples_leaf=4, n_jobs=-1)
# # rgrs = ExtraTreesRegressor(n_estimators=500, max_features=15, max_depth=15, min_samples_leaf=4, n_jobs=-1)
# rgrs.fit(train, train_y)
# pred = rgrs.predict(test)

# #generate solution
# preds = pd.DataFrame({"Id": test_ind, "Hazard": pred})
# preds = preds.set_index('Id')
# preds.to_csv('output_rf_0.5.csv')

##################################################################################
# trees.rgrs.feature_importances_
# Out[11]: 
# array([ 0.07293952,  0.07345944,  0.04040399,  0.03368627,  0.02372594,
#         0.0119912 ,  0.00609134,  0.05568966,  0.01950578,  0.02756495,
#         0.03277017,  0.0196826 ,  0.02171153,  0.01458485,  0.01324964,
#         0.04697818,  0.00519976,  0.08796772,  0.05909358,  0.01048869,
#         0.05411541,  0.0161813 ,  0.0152753 ,  0.03459764,  0.00174145,
#         0.06034286,  0.03411409,  0.007873  ,  0.00441298,  0.02288045,
#         0.02887455,  0.04280616])

# columns = np.array(columns)
# rf_dict = {}
# for i in range(len(rf_i)):
#     rf_dict[columns[i]] = rf_i[i]
# rf_sorted = sorted(rf_dict.items(), key=lambda x:x[1], reverse=True)

# et_dict = {}
# for i in range(len(et_i)):
#     et_dict[columns[i]] = et_i[i]
# et_sorted = sorted(et_dict.items(), key=lambda x:x[1], reverse=True)

# trees.rf.feature_importances_
# Out[16]: 
# array([ 0.06040579,  0.06494917,  0.04318726,  0.02615279,  0.01852012,
#         0.01168841,  0.01191699,  0.02960871,  0.01511068,  0.03240706,
#         0.02806397,  0.01816361,  0.02649339,  0.01744862,  0.01160299,
#         0.04006211,  0.00531746,  0.10046412,  0.06620769,  0.01162953,
#         0.06317604,  0.01827922,  0.01930202,  0.039515  ,  0.00360228,
#         0.06725282,  0.04031323,  0.0076548 ,  0.00495549,  0.0217165 ,
#         0.0321081 ,  0.04272402])

# [('T2_V1', 0.10046412383770369),
#  ('T2_V9', 0.067252818230216449),
#  ('T2_V2', 0.0662076868935767),
#  ('T1_V2', 0.064949165644358917),
#  ('T2_V4', 0.06317604252601923),
#  ('T1_V1', 0.06040579435074165),
#  ('T1_V3', 0.043187264459862262),
#  ('T2_V15', 0.042724016361761021),
#  ('T2_V10', 0.040313231300572135),
#  ('T1_V16', 0.040062114239650344),
#  ('T2_V7', 0.039515003399529298),
#  ('T1_V10', 0.032407062391910095),
#  ('T2_V14', 0.03210809815025336),
#  ('T1_V8', 0.02960870993830976),
#  ('T1_V11', 0.028063965890809856),
#  ('T1_V13', 0.026493386172870132),
#  ('T1_V4', 0.026152794260721049),
#  ('T2_V13', 0.021716498973064175),
#  ('T2_V6', 0.019302019652019702),
#  ('T1_V5', 0.018520122533330234),
#  ('T2_V5', 0.018279216466744103),
#  ('T1_V12', 0.018163609872469434),
#  ('T1_V14', 0.017448618744810348),
#  ('T1_V9', 0.0151106819624909),
#  ('T1_V7', 0.011916987625035546),
#  ('T1_V6', 0.011688413875196995),
#  ('T2_V3', 0.01162953407565671),
#  ('T1_V15', 0.011602990917997064),
#  ('T2_V11', 0.0076547988500881934),
#  ('T1_V17', 0.0053174590536080712),
#  ('T2_V12', 0.0049554889418797661),
#  ('T2_V8', 0.0036022804067429009)]

# trees.et.feature_importances_
# Out[17]: 
# array([ 0.05007258,  0.05074681,  0.04240309,  0.03274856,  0.02554925,
#         0.02490933,  0.01755993,  0.03344396,  0.02120232,  0.03708632,
#         0.03325449,  0.022544  ,  0.03592348,  0.02633651,  0.01709456,
#         0.04089661,  0.01062673,  0.04959016,  0.04607433,  0.02454109,
#         0.04220765,  0.02446572,  0.02881448,  0.03959946,  0.00626998,
#         0.04476032,  0.03978942,  0.0173371 ,  0.00937625,  0.02914092,
#         0.03675593,  0.03887866])
# [('T1_V2', 0.050746808629645257),
#  ('T1_V1', 0.050072583946285533),
#  ('T2_V1', 0.049590163358946233),
#  ('T2_V2', 0.046074326855031211),
#  ('T2_V9', 0.044760317933895385),
#  ('T1_V3', 0.042403086182007881),
#  ('T2_V4', 0.042207645886217261),
#  ('T1_V16', 0.040896606728604359),
#  ('T2_V10', 0.039789421889369049),
#  ('T2_V7', 0.039599459210547604),
#  ('T2_V15', 0.038878658490802888),
#  ('T1_V10', 0.037086323217594114),
#  ('T2_V14', 0.036755931739512873),
#  ('T1_V13', 0.035923483356079373),
#  ('T1_V8', 0.033443956927383965),
#  ('T1_V11', 0.033254489164736549),
#  ('T1_V4', 0.032748559564775147),
#  ('T2_V13', 0.029140917541681501),
#  ('T2_V6', 0.028814482375866632),
#  ('T1_V14', 0.026336512025995962),
#  ('T1_V5', 0.02554924583730123),
#  ('T1_V6', 0.024909327153720091),
#  ('T2_V3', 0.024541087733418912),
#  ('T2_V5', 0.024465722321089785),
#  ('T1_V12', 0.022543997195630307),
#  ('T1_V9', 0.021202318740287218),
#  ('T1_V7', 0.017559934840931921),
#  ('T2_V11', 0.017337104244391232),
#  ('T1_V15', 0.017094555296359247),
#  ('T1_V17', 0.010626733747732547),
#  ('T2_V12', 0.0093762544193892374),
#  ('T2_V8', 0.0062699834447697105)]

