# import pandas as pd
# import numpy as np 
# import pickle
# from sklearn import preprocessing

# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import RandomizedSearchCV
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neighbors import KNeighborsClassifier

# ##################################################################################
# # cal metric

# def gini(solution, submission):
#     df = zip(solution, submission, range(len(solution)))
#     df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
#     rand = [float(i+1)/float(len(df)) for i in range(len(df))]
#     totalPos = float(sum([x[0] for x in df]))
#     cumPosFound = [df[0][0]]
#     for i in range(1,len(df)):
#         cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
#     Lorentz = [float(x)/totalPos for x in cumPosFound]
#     Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
#     return sum(Gini)

# def normalized_gini(solution, submission):
#     normalized_gini = gini(solution, submission)/gini(solution, solution)
#     return normalized_gini

# ##################################################################################
# #load train and test 

# train  = pd.read_csv('./data/train.csv', index_col=0)
# test  = pd.read_csv('./data/test.csv', index_col=0)
# train_y = train.Hazard

# # drop train_y -> train_y
# train.drop('Hazard', axis=1, inplace=True)
# # drop noisy features
# train.drop('T2_V10', axis=1, inplace=True)
# train.drop('T2_V7', axis=1, inplace=True)
# train.drop('T1_V13', axis=1, inplace=True)
# train.drop('T1_V10', axis=1, inplace=True)

# test.drop('T2_V10', axis=1, inplace=True)
# test.drop('T2_V7', axis=1, inplace=True)
# test.drop('T1_V13', axis=1, inplace=True)
# test.drop('T1_V10', axis=1, inplace=True)

# # columns and index for later use
# columns = train.columns
# test_ind = test.index

# train = np.array(train)
# test = np.array(test)

# # label encode the categorical variables
# for i in range(train.shape[1]):
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(train[:,i]) + list(test[:,i]))
#     train[:,i] = lbl.transform(train[:,i])
#     test[:,i] = lbl.transform(test[:,i])

# train = train.astype(np.float32)
# test = test.astype(np.float32)

# ##################################################################################

# with open('./data/train_denoise.vec', 'rb') as f:
#     train = pickle.load(f)

# with open('./data/test_denoise.vec', 'rb') as f:
#     test = pickle.load(f)

# with open('./data/train_y.vec', 'rb') as f:
#     train_y = pickle.load(f)

# train_x_sp, test_x_sp, train_y_sp, test_y_sp = train_test_split(train, train_y, train_size=0.8, random_state=50)

# rgrs = KNeighborsRegressor(n_neighbors =100)
# rgrs.fit(train_x_sp, train_y_sp)
# pred = rgrs.predict(test_x_sp)

# score = normalized_gini(test_y_sp, pred)
# print '{:.6f}'.format(score)
