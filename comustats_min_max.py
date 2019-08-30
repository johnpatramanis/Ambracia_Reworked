import numpy as np

# create dictionary for stats
COMUS_FILE=open('COMUSTATS_1','r')
parameters={}
for line in COMUS_FILE:
    line=line.strip().split()
    for k in range(0,len(line)):
        parameters[k]=[]

for x in range(0,290000):
    COMUS_FILE=open('COMUSTATS_{}'.format(x),'r')
    j=0
    COMUS_FILE.readline()
    for line in COMUS_FILE:
        line=line.strip().split()
        for k in range(0,len(line)):
            parameters[k].append(float(line[k]))
    for y in range(0,len(parameters)):
        parameters[y]=[min(parameters[y]),max(parameters[y])]
    print(parameters)

COMUS_MIN_MAX=open('COMUS_MIN_MAX','w')
for w in range(0,len(parameters)):
    min=str(parameters[w][0])
    max=str(parameters[w][1])
    COMUS_MIN_MAX.write(min+'\t'+max+'\n')
