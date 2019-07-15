import numpy as np
import matplotlib.pyplot as plt
import math 

rep=1
for rep in range(1,14):
    file=open('PREDICTIONS_MEAN_{}'.format(rep),'r')
    actual=[]
    predicted_mean=[]
    predicted_med=[]
    predicted_mode=[]
    
    for line in file:
        line=line.strip().split()
        actual.append(float(line[0]))
        predicted_med.append(float(line[1]))
        predicted_mean.append(float(line[2]))
        predicted_mode.append(float(line[3]))

    print(min(actual),min(predicted_mean),min(predicted_med),min(predicted_mode))
    print(max(actual),max(predicted_mean),max(predicted_med),max(predicted_mode))

    RMSE_MEAN=[]
    RMSE_MED=[]
    RMSE_MODE=[]

    #Root Mean Squared Error 
    for k in range(0,len(actual)):
        RMSE_MEAN.append((actual[k]-predicted_mean[k])**2)
        RMSE_MED.append((actual[k]-predicted_med[k])**2)
        RMSE_MODE.append((actual[k]-predicted_mode[k])**2)



    ROOT=math.sqrt((sum(RMSE_MEAN))/len(actual))
    RMSE_MEAN=ROOT/(max(actual)-min(actual))
    
    ROOT=math.sqrt((sum(RMSE_MED))/len(actual))
    RMSE_MED=ROOT/(max(actual)-min(actual))
    
    ROOT=math.sqrt((sum(RMSE_MODE))/len(actual))
    RMSE_MODE=ROOT/(max(actual)-min(actual))
    
    print('The Root Mean Squared Error for parameter number {} is {}(mean) {}(med) {}(mode) '.format(rep,RMSE_MEAN,RMSE_MED,RMSE_MODE))
