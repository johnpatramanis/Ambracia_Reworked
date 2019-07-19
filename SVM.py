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

DATAFILE=open('FOR_ABC','r')


LABELS=[]
DATA=[]
for line in DATAFILE:
    line=line.strip().split()
    LABELS.append(int(line[0]))
    DATA.append([float(x) for x in line[14:]])
    
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