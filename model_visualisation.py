import msprime
import argparse
import numpy as np
import math
import os
import time
import re
import random
import numba
import umap

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import TapTool, CustomJS, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.io import output_file
from bokeh import colors



def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


def get_colors(n):
    colorz=[]
    for k in range(0,n):
        c1=round(random.random(), 3)
        c2=round(random.random(), 3)
        c3=round(random.random(), 3)
        colorz.append((c1,c2,c3))
    return colorz
#####IIIINCOOOMPLEEEETE

@numba.jit(nopython=False)
def weighted_dist(a,b):
    return math.sqrt(sum([((a[x] - b[x])*WEIGHTS[x]) ** 2 for x in range(0,len(a))]))
    
def reject_outliers(data, m=2):
    MEAN=np.mean(data,axis=0)
    MEAN_DIST=[]
    for k in data:
        MEAN_DIST.append(np.abs(np.linalg.norm(k-MEAN)))
    MEAN_DIST=np.median(MEAN_DIST)
    print(MEAN_DIST)
    
    CLEANDATA=[]
    for k in data:
        if np.abs(np.linalg.norm(k-MEAN))<=MEAN_DIST*2:
            CLEANDATA.append(k)  
    
    
    
    return np.asarray(CLEANDATA)
    
    
FILE_DIST=open('FOR_ABC','r')
FILE_MEANS=open('FOR_SVM','r')



DATA=[]
for line in FILE_DIST:
    line=line.strip().split()
    for x in range(0,len(line)):
        line[x]=float(line[x])
    DATA.append(line)
    if len(DATA)>=1000:
        break
LABELS=[x[0] for x in DATA]
SUMSTATS=[x[14:] for x in DATA]

print(len(LABELS),len(SUMSTATS))

X = np.asarray(SUMSTATS)  
PCAvis = PCA(n_components=2).fit_transform(X)
PCAvis=reject_outliers(PCAvis,m=2)
PCAvis=reject_outliers(PCAvis,m=2)
PCAvis=reject_outliers(PCAvis,m=2)
print(PCAvis.shape,PCAvis)

print(PCAvis)

colors=[]
for x in LABELS:
    if x==1:
        colors.append('red')
    else:
        colors.append('blue')



plt.scatter([x[0] for x in PCAvis],[x[1] for x in PCAvis],c=colors)
plt.show()











