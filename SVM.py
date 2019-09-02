# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:58:03 2018

@author: John Patramanis,original code from aaggelos
"""

import math
import numpy as np
from numpy import genfromtxt
from numpy import random
import timeit
import matplotlib.pyplot as plt
#import pandas as pd #not sure if needed
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



###-------------------------- Random Forests --------------------------
print("Random Forest")


pipeline_rf = make_pipeline(StandardScaler(), RandomForestClassifier( n_jobs=10))

crit = ['gini', 'entropy']
numt = [100, 150] #number of trees
feats = ['log2','sqrt'] #max_features considered during each split
depth = [ 20, 30, None] #maximum depth each tree is allowed to reach
# 
hyperparameters = {#'randomforestclassifier__criterion' : crit,
                   'randomforestclassifier__n_estimators' : numt,
                   'randomforestclassifier__max_features' : feats,
                   'randomforestclassifier__max_depth': depth}


nested_score = np.zeros([60, 1]).astype(int)
print(nested_score.shape)

inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

## Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(pipeline_rf, param_grid=hyperparameters, cv=inner_cv)

set_scores = []

#for each demographic pair we perform nested cross validation

for i in range(1, 61):
    kappa = cross_val_score(clf, X=DATA, y=LABELS, cv=outer_cv, n_jobs=10).mean()
    nested_score[i-1,0] = kappa
    set_scores.append(kappa)
    print("dataset" + str(i))
with open ("forest_res.txt",'w') as rf:
    for j in range (0, 60):
        rf.write(str(set_scores[j]))
    
print(kappa)

print(set_scores)

#############################################################################################################################################################
#############################################################################################################################################################

nested_scores = {}

clf_names = ['Corinthian Origin', 'Local Origin']

true_pos = []
false_pos = []

accuracies = np.zeros((61,3)).astype(float)
print(accuracies.shape)

dtst = 0
def get_the_rep(y_true, y_pred):
    global dtst
    nested_scores.update({'params'+str(dtst) : classification_report(y_pred,y_true,target_names = clf_names)})
    
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=0)
    true_pos.append(tpr)
    false_pos.append(fpr)         
    
    
    dtst = dtst + 1
    return accuracy_score(y_true, y_pred)


#def get_the_rep(y_true, y_pred):
#    global dtst
#    best_params.update({'params'+str(dtst) : classification_report(y_pred,y_true,target_names = clf_names)})
#    dtst = dtst + 1
#    return accuracy_score(y_true, y_pred)



#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

#------------------ Support Vector Machines ---------------
print("Support Vector Machines")
pipeline_svm = make_pipeline(StandardScaler(), SVC())
cst = [1, 2, 5, 7, 10] #starting from hard-margin and loosening to see performance
ker = ['poly']
deg = [1] #just for the polynomial kernel


hyperparameters = { 'svc__C' : cst,
                   'svc__kernel': ker,
                   'svc__degree': deg,
                   }

inner_cv = KFold(n_splits=10, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

## Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(pipeline_svm, param_grid=hyperparameters, cv=inner_cv, n_jobs=-1)



#for each demographic pair we perform nested cross validation
for feats in range (40, 41):
    
    set_scores_svm = []
 #   global true_pos, false_pos, nested_scores

    nested_scores = {}
    true_pos = []
    false_pos = []
    
    print (feats)
    for i in range(1, num_sets):
        print(i)
        data = np.concatenate((neut_data['neut'+str(i)],slct_data['sel'+str(i)]), axis = 0)
        selected = SelectKBest(score_func=mutual_info_classif, k=feats).fit(data, labels[:,0])
        current = selected.transform(data)
        
        kappa = cross_val_score(clf, X=current, y=labels[:,0], cv=outer_cv, scoring = make_scorer(get_the_rep)).mean()
        set_scores_svm.append(kappa)    
    
    with open ("split_res/acc/tot_acc_split" + str(feats) + ".txt", "w") as rf:
        for j in range (0, 60):
           rf.write(str(set_scores_svm[j]) + "\n")
    
    np.savetxt("split_res/tpr/tpr_split" + str(feats) + ".txt", np.array(true_pos[:]),  fmt = '%1.8f')
    np.savetxt("split_res/fpr/fpr_split" + str(feats) + ".txt",  np.array(false_pos[:]),  fmt = '%1.8f' )
    #np.savetxt("split_res/nested/rep_split" + str(feats) + ".txt", np.array(true_pos[:]),  fmt = '%1.8f')
    	#rp.write(str(nested_scores))
