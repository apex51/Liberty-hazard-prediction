# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, mutual_info_score


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

# The competition datafiles are in the directory ../input
# Read competition data files
train  = pd.read_csv('./data/train.csv', index_col=0)
test  = pd.read_csv('./data/test.csv', index_col=0)

train_y = np.array(train.Hazard)

y_01 = np.array(train_y) # 0 stands for num other than 1
y_110 = np.array(train_y) # num 1-10

for i in range(len(train_y)):
    if train_y[i] != 1:
        y_01[i] = 0

for i in range(len(train_y)):
    if train_y[i] > 11:
        y_110[i] = 10
    if train_y[i] ==10 or train_y[i] ==11:
        y_110[i] = 9
    if train_y[i] ==9 or train_y[i] ==8:
        y_110[i] = 8



train.drop('Hazard', axis=1, inplace=True)
# drop noisy features
# train.drop('T2_V10', axis=1, inplace=True)
# train.drop('T2_V7', axis=1, inplace=True)
# train.drop('T1_V13', axis=1, inplace=True)
# train.drop('T1_V10', axis=1, inplace=True)

# test.drop('T2_V10', axis=1, inplace=True)
# test.drop('T2_V7', axis=1, inplace=True)
# test.drop('T1_V13', axis=1, inplace=True)
# test.drop('T1_V10', axis=1, inplace=True)

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
train_y = np.array(train_y).astype(float)

mi = []
nmi = []
ami = []

for i in range(train.shape[1]):
    mi = np.append(mi, mutual_info_score(train[:,i],y_110))
    nmi = np.append(nmi, normalized_mutual_info_score(train[:,i],y_110))
    ami = np.append(ami, adjusted_mutual_info_score(train[:,i],y_110))

columns = np.array(columns)
mi_dict = {}
for i in range(train.shape[1]):
    mi_dict[columns[i]] = mi[i]
mi_sorted = sorted(mi_dict.items(), key=lambda x:x[1], reverse=True)

nmi_dict = {}
for i in range(train.shape[1]):
    nmi_dict[columns[i]] = nmi[i]
nmi_sorted = sorted(nmi_dict.items(), key=lambda x:x[1], reverse=True)

ami_dict = {}
for i in range(train.shape[1]):
    ami_dict[columns[i]] = ami[i]
ami_sorted = sorted(ami_dict.items(), key=lambda x:x[1], reverse=True)



# mutualinfo.mi_sorted
# Out[26]: 
# [('T2_V1', 0.031147100649547579),
#  ('T1_V2', 0.024465552401531223),
#  ('T1_V1', 0.023158183645566932),
#  ('T1_V16', 0.020706899187687106),
#  ('T1_V11', 0.015982035759597023),
#  ('T1_V5', 0.014991416208208763),
#  ('T2_V2', 0.013131915080916694),
#  ('T1_V9', 0.012206911899635926),
#  ('T2_V4', 0.011247896606553225),
#  ('T2_V9', 0.011097655583233224),
#  ('T1_V15', 0.011002709708234581),
#  ('T1_V8', 0.010243743322038108),
#  ('T1_V4', 0.0099299616680986191),
#  ('T2_V15', 0.0094143281159686718),
#  ('T1_V3', 0.0061287116928783886),
#  ('T1_V12', 0.0052997312533250062),
#  ('T1_V14', 0.0046857258504557405),
#  ('T2_V5', 0.0041253235406875279),
#  ('T2_V14', 0.0031745675426102203),
#  ('T2_V6', 0.0031176849921766272),
#  ('T2_V13', 0.0027887368391461918),
#  ('T2_V10', 0.0026868871430718267),
#  ('T2_V7', 0.0025048705199930367),
#  ('T1_V7', 0.0023112995468243473),
#  ('T1_V10', 0.0017678604341222208),
#  ('T1_V13', 0.0013087553066160368),
#  ('T1_V6', 0.00077361687366338196),
#  ('T2_V12', 0.00075595063388385602),
#  ('T1_V17', 0.0006720635473672213),
#  ('T2_V8', 0.00064546165473161024),
#  ('T2_V3', 0.00056602110351295981),
#  ('T2_V11', 0.0004721949277378091)]


# mutualinfo.nmi_sorted
# Out[27]: 
# [('T1_V8', 0.01111324976686766),
#  ('T1_V15', 0.010701290782880325),
#  ('T2_V1', 0.010084328244128948),
#  ('T1_V2', 0.0095951858875745026),
#  ('T1_V16', 0.0093405287460753116),
#  ('T1_V1', 0.0093041533148907375),
#  ('T1_V11', 0.0084298588606901842),
#  ('T1_V9', 0.0080851150748477082),
#  ('T1_V5', 0.0078352722205443162),
#  ('T1_V12', 0.0059958656778946341),
#  ('T1_V4', 0.0056603817663048077),
#  ('T2_V2', 0.0053362558948363923),
#  ('T2_V15', 0.0045925691304468637),
#  ('T2_V4', 0.0045100433379651689),
#  ('T2_V9', 0.004306051635776258),
#  ('T1_V14', 0.0033869620866894953),
#  ('T1_V3', 0.0030799681453686188),
#  ('T1_V7', 0.0030373115557743328),
#  ('T2_V5', 0.0029480261165630924),
#  ('T2_V6', 0.0021072270196103868),
#  ('T2_V14', 0.0017869760762807931),
#  ('T2_V13', 0.0017361490165045896),
#  ('T2_V10', 0.0013450653073497686),
#  ('T2_V7', 0.0012645277039456558),
#  ('T2_V8', 0.0011793013397838131),
#  ('T1_V10', 0.00097862254542784174),
#  ('T1_V13', 0.00079639816770553655),
#  ('T2_V12', 0.00075215320831321388),
#  ('T1_V17', 0.00065839579336819329),
#  ('T1_V6', 0.0006404040828904712),
#  ('T2_V3', 0.00048939567638624627),
#  ('T2_V11', 0.00042190284796834304)]


#  mutualinfo.ami_sorted
# Out[28]: 
# [('T1_V16', 0.0063588119303101967),
#  ('T1_V11', 0.0056778647005344242),
#  ('T1_V5', 0.0056476700879033205),
#  ('T1_V1', 0.00553696225616812),
#  ('T1_V2', 0.0052072041224359468),
#  ('T1_V9', 0.00485959108606881),
#  ('T1_V8', 0.0043000030095684842),
#  ('T1_V15', 0.0041714816558051735),
#  ('T1_V4', 0.0033738154268953663),
#  ('T2_V15', 0.0023987552413936852),
#  ('T1_V12', 0.0019827800005949659),
#  ('T1_V14', 0.0015021011771098342),
#  ('T1_V3', 0.0013675945509428529),
#  ('T2_V4', 0.0011759222187234528),
#  ('T2_V5', 0.0011430384441659832),
#  ('T2_V2', 0.00085481203908116955),
#  ('T2_V9', 0.00068501046972756512),
#  ('T1_V7', 0.00062264454997770639),
#  ('T2_V13', 0.00049793099460795106),
#  ('T2_V6', 0.0004768349155311371),
#  ('T2_V1', 0.00041635806694417587),
#  ('T2_V14', 0.00033208864052998975),
#  ('T2_V12', 0.0001219840576045087),
#  ('T1_V6', 0.00010568677083522096),
#  ('T1_V17', 8.0625185551207626e-05),
#  ('T2_V8', 1.4747709122219557e-05),
#  ('T2_V3', 1.4302635258957957e-05),
#  ('T2_V10', -1.7160029022562435e-05),
#  ('T2_V11', -2.5294228065578913e-05),
#  ('T1_V10', -6.3672857564379693e-05),
#  ('T1_V13', -6.6326899045625641e-05),
#  ('T2_V7', -9.6906482539386334e-05)]


# for 1-11 transform
# mutualinfo.ami_sorted
# Out[19]: 
# [('T1_V16', 0.0062889859141909109),
#  ('T1_V11', 0.0061868965610259695),
#  ('T1_V5', 0.0060045214642763375),
#  ('T1_V9', 0.0053609458806294363),
#  ('T1_V1', 0.0050880693978074022),
#  ('T1_V2', 0.0049468584464248278),
#  ('T1_V15', 0.0045786742113630421),
#  ('T1_V8', 0.0038061049489044548),
#  ('T1_V4', 0.0037076895485162599),
#  ('T2_V15', 0.0023558009630067788),
#  ('T1_V12', 0.0020504431545598726),
#  ('T1_V14', 0.0017052726186459022),
#  ('T1_V3', 0.0015239835507215244),
#  ('T2_V5', 0.0012209823961438495),
#  ('T2_V4', 0.001082901221946043),
#  ('T2_V9', 0.00077005329768889256),
#  ('T2_V2', 0.00071163653702712392),
#  ('T1_V7', 0.00062152875011316905),
#  ('T2_V13', 0.00051897203562512116),
#  ('T2_V6', 0.00045255563204305461),
#  ('T2_V14', 0.00035328960707335476),
#  ('T2_V1', 0.00012908991150293247),
#  ('T2_V12', 9.1441442091612959e-05),
#  ('T1_V6', 3.1842001497535953e-05),
#  ('T2_V8', 3.0651297404160979e-05),
#  ('T1_V17', 2.9120738058502276e-05),
#  ('T1_V13', -3.9592786488369736e-06),
#  ('T2_V11', -1.4455071606004407e-05),
#  ('T1_V10', -1.912064811507346e-05),
#  ('T2_V3', -2.2686252820854369e-05),
#  ('T2_V7', -6.2219659684669475e-05),
#  ('T2_V10', -6.8210463321557944e-05)]