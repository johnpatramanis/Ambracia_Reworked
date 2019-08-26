import msprime
import numpy as np
import math
import os
import argparse
import time
import re
import random
import sys
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

DATAFILE=open('FOR_ABC','r')


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

RANDOM_INTS=np.random.choice(len(DATA),len(DATA)/10,replace=False)

TEST_DATA=[DATA[x] for x in range(0,len(DATA)) if x in RANDOM_INTS]
TEST_LABELS=[LABELS[x] for x in range(0,len(LABELS)) if x in RANDOM_INTS]

DATA=[DATA[x] for x in range(0,len(DATA)) if x not in RANDOM_INTS]
LABELS=[LABELS[x] for x in range(0,len(LABELS)) if x not in RANDOM_INTS]

print(len(DATA),len(TEST_DATA),len(LABELS),len(TEST_LABELS))

clf = svm.SVC(gamma='scale')
clf.fit(DATA,LABELS)
predictions=clf.predict(TEST_DATA)
CORRECTNESS=[]
for x in range(0,len(predictions)):
    if predictions[x]!=TEST_LABELS[x]:
        CORRECTNESS.append(0)
    if predictions[x]==TEST_LABELS[x]:
        CORRECTNESS.append(1)
print(np.sum(CORRECTNESS)/len(CORRECTNESS))




#to be implemented
#feature selection
clf = LassoCV(cv=5)

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()





####
#model selection

from sklearn.pipeline import Pipeline
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)

parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}

grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)

print (grid.best_params_)