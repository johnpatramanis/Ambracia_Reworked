# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:58:03 2018

@author: John Patramanis,original code from aaggelos
"""

import math
import random
import numpy as np
from numpy import genfromtxt
from numpy import random
import timeit
import matplotlib.pyplot as plt
import pandas as pd 
#model selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#general
import math
import numpy as np
from numpy import genfromtxt
import timeit
import matplotlib.pyplot as plt
import re

#prediction

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve


#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

#how to crash a server 101
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing


start_time = timeit.default_timer()



DATAFILE=open('FOR_SVM','r')


LABELS=[]
DATA=[]
for line in DATAFILE:
    line=line.strip().split()
    LABELS.append(int(line[0]))
    DATA.append([float(x) for x in line[14:]])
    
X=DATA
Y=LABELS
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)
    
print(len(LABELS),len(DATA))

randints=np.random.randint(0,len(DATA),size=1000,dtype='int')
print(randints)


Final_data=[DATA[x] for x in randints]
Final_labels=[LABELS[x] for x in randints]

DATA=[DATA[x] for x in range(0,len(DATA)) if x not in randints]
LABELS=[LABELS[x] for x in range(0,len(LABELS)) if x not in randints]

print(len(Final_data),len(Final_labels))
print(len(DATA),len(LABELS))



###-------------------------- Random Forests --------------------------
print("Random Forest")


pipeline_rf = make_pipeline(StandardScaler(),VarianceThreshold(threshold=(.8 * (1 - .8))), RandomForestClassifier( n_jobs=16))

crit = ['gini', 'entropy']
numt = [100, 150,300,500,1000] #number of trees
feats = ['log2','sqrt'] #max_features considered during each split
depth = [ 20, 30,40,50,None] #maximum depth each tree is allowed to reach
#variances=[.8*(1 - .8),]


# 
hyperparameters = {#'randomforestclassifier__criterion' : crit,
                   'randomforestclassifier__n_estimators' : numt,
                   'randomforestclassifier__max_features' : feats,
                   #'VarianceThreshold__max_features' : feats,
                   'randomforestclassifier__max_depth': depth}




inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

## Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(pipeline_rf, param_grid=hyperparameters, cv=inner_cv)

clf.fit(DATA,LABELS)

#for each demographic pair we perform nested cross validation

print(pd.DataFrame.from_dict(clf.cv_results_))
print('#######################################')
print(clf.best_estimator_)
print(clf.best_params_)

print('#######################################')

predictedd=clf.predict(Final_data)
results=open('results.txt.','w')
print(predictedd)
print(Final_labels)
Predictions=[]
for k in range(0,len(predictedd)):
    results.write(str(predictedd[k]))
    results.write('\t')
    results.write(str(Final_labels[k]))
    results.write('\n')
    if str(predictedd[k])==str(Final_labels[k]):
        Predictions.append(1)
    else:
        Predictions.append(0)

print(np.mean(Predictions))
results.write(str(np.mean(Predictions)))
#print(clf.cv_results_[clf.best_index_])
kappa = cross_val_score(clf, X=DATA, y=LABELS, cv=outer_cv, n_jobs=12)


print(kappa)

#print(set_scores)
print('finished')
#############################################################################################################################################################
##################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

#------------------ Support Vector Machines ---------------

#------------------ Support Vector Machines ---------------
print("Support Vector Machines")
pipeline_svm = make_pipeline(StandardScaler(),SVC())
cst = [1, 2, 5, 7, 10] #starting from hard-margin and loosening to see performance
ker = ['poly']
deg = [1] #just for the polynomial kernel


hyperparameters = { 'svc__C' : cst,
                   'svc__kernel': ker,
                   'svc__degree': deg,
                   }

inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

print('CV')
print(outer_cv)

## Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(pipeline_svm, param_grid=hyperparameters, cv=inner_cv, n_jobs=12)


set_scores_svm=[]
FEATS=['all',20,30,50]

#   global true_pos, false_pos, nested_scores
for feats in FEATS:
    selected = SelectKBest(score_func=mutual_info_classif, k=feats).fit(DATA, LABELS)
    current = selected.transform(DATA)

    kappa = cross_val_score(clf, X=current, y=LABELS, cv=outer_cv).mean()
    set_scores_svm.append(kappa)
    print(kappa)
print(set_scores_svm)
