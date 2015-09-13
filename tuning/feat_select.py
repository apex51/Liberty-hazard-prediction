# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RandomizedLasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.cross_validation import train_test_split

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

def make_dummies(df, variables):
    for variable in variables:
        dummies = pd.get_dummies(df[variable], prefix = variable)
        df = pd.concat([df, dummies], axis = 1)
        df = df.drop(variable, 1)
    return df

def main():
    print "read train"
    df_train = pd.read_csv('./data/train.csv')
    print "read test"
    df_test = pd.read_csv('./data/test.csv')
    sample = pd.read_csv('./data/sample_submission.csv')
    
    cats = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 
            'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16',
            'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',
            'T2_V13']
            
    print "convert mixed columns to strings"
    df_train.loc[:, cats] = df_train[cats].applymap(str)
    df_test.loc[:, cats] = df_test[cats].applymap(str)
    
    print "one-hot encoding"
    df_train = make_dummies(df_train, cats)
    df_test = make_dummies(df_test, cats)
    
    print "set binary labels"
    df_train['hazard_class'] = (df_train.Hazard==1).astype(int)
    
    classes = df_train.hazard_class.values
    # loss = df_train.target.values
    hazard = df_train.Hazard.values
    df_train = df_train.drop(['Hazard', 'Id', 'hazard_class'], axis = 1)
    df_test = df_test.drop(['Id'], axis = 1)

    build_features = False #flag, determines whether features will be trained or read from file
    
    if build_features:
        print "univariate feature selectors"
        selector_clf = SelectKBest(score_func = f_classif, k = 'all')
        selector_reg = SelectKBest(score_func = f_regression, k = 'all')
        selector_clf.fit(df_train.values, classes)
        selector_reg.fit(df_train.values, hazard)
        pvalues_clf = selector_clf.pvalues_
        pvalues_reg = selector_reg.pvalues_
        pvalues_clf[np.isnan(pvalues_clf)] = 1
        pvalues_reg[np.isnan(pvalues_reg)] = 1
        
        #put feature vectors into dictionary
        feats = {}
        feats['univ_sub01'] = (pvalues_clf<0.1)&(pvalues_reg<0.1) 
        feats['univ_sub005'] = (pvalues_clf<0.05)&(pvalues_reg<0.05)
        feats['univ_reg_sub005'] = (pvalues_reg<0.05)
        feats['univ_clf_sub005'] = (pvalues_clf<0.05)
        
        print "randomized lasso feature selector"
        sel_lasso = RandomizedLasso(random_state = 42).fit(df_train.values, hazard)
        #put rand_lasso feats into feature dict
        feats['rand_lasso'] = sel_lasso.get_support()
        
        print "l1-based feature selectors"
        X_sp = sparse.coo_matrix(df_train.values)
        sel_svc = LinearSVC(C=0.1, penalty = "l1", dual = False, random_state = 42).fit(X_sp, classes)
        feats['LinearSVC'] = np.ravel(sel_svc.coef_>0)
        sel_log = LogisticRegression(C=0.01, random_state = 42).fit(X_sp, classes)
        feats['LogReg'] = np.ravel(sel_log.coef_>0)
        
        feat_sums = np.zeros(len(feats['rand_lasso']))
        for key in feats:
            feat_sums+=feats[key].astype(int)
        feats['ensemble'] = feat_sums>=5 #take features which get 5 or more votes
        joblib.dump(feats, './features/feats.pkl', compress = 3)
    
    else:
        feats = joblib.load('features/feats.pkl')

    xtrain = df_train.values
    xtest = df_test.values

    print "fitting xgb-regressor"
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["max_depth"] = 7
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["min_child_weight"] = 5
    params["silent"] = 1
    plst = list(params.items())
    num_rounds = 600
    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(xtrain[:,feats['ensemble']], label=hazard)
    xgtest = xgb.DMatrix(xtest[:,feats['ensemble']])
    reg_xgb = xgb.train(plst, xgtrain, num_rounds)
    xgb_preds = reg_xgb.predict(xgtest)
    sample['Hazard'] = xgb_preds
    sample.to_csv('./submissions/xgb.csv', index = False)
    reg_lin = LinearRegression()
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    print "fitting linear regressor"
    reg_lin.fit(xtrain[:, feats['rand_lasso']], hazard)
    lin_preds = reg_lin.predict(xtest[:, feats['rand_lasso']])
    sample['Hazard'] = lin_preds
    sample.to_csv('./submissions/lin.csv', index = False)
    xgb_order = xgb_preds.argsort().argsort() #maps smallest value to 0, second-smallest to 1 etc.
    lin_order = lin_preds.argsort().argsort()
    #averaging
    mean_order = np.vstack((xgb_order, lin_order)).mean(0)    
    sample['Hazard'] = mean_order
    sample.to_csv('./submissions/mean.csv', index = False)

if __name__ == '__main__':
    main()