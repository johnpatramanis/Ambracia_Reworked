import msprime
import numpy as np
import numpy.linalg
import math
import os
import time
import re
import random
import sys
import os.path
from multiprocessing import Process,Manager




start_time = time.time()


reps=1
for REPS in range(0,reps):

 
    parametersfile=open('PARAMETERS_{}'.format(REPS),'w')
    
    N_locals=int(round(random.uniform(300.0,4000.0)))
    N_metropolis=int(round(random.uniform(300.0,4000.0)))
    
    generation_time = 20
    T_COLONIZATION=700/generation_time
    
    
    COLONIZER=random.randint(0,1)
    if COLONIZER==0:
        N_initial_colony=int(round(random.uniform(100.0,float(N_locals))))
        while N_initial_colony>N_metropolis:
            N_initial_colony=int(round(random.uniform(100.0,float(N_metropolis))))
    if COLONIZER==1:
        N_initial_colony=int(round(random.uniform(100.0,float(N_metropolis))))



    r_locals=10**(-1*random.uniform(1,4))
    r_metropolis=10**(-1*random.uniform(1,4))
    r_colony=10**(-1*random.uniform(1,4))
    
    growth_counter=0
    while (float(N_initial_colony) / (math.exp(-r_colony * T_COLONIZATION)) ) > float(N_metropolis):
        r_colony=10**(-1*random.uniform(1,4))
        growth_counter+=1
        if growth_counter>=1000000:
            r_colony=0
            break

    N_finale_colony=N_initial_colony / (math.exp(-r_colony * T_COLONIZATION))
    print(N_locals,N_metropolis,N_initial_colony,N_finale_colony)


    migration_matrix = [
        [0,10**(-1*random.uniform(1,4)),10**(-1*random.uniform(1,4))],
        [10**(-1*random.uniform(1,4)),0,10**(-1*random.uniform(1,4))],
        [10**(-1*random.uniform(1,4)),10**(-1*random.uniform(1,4)),0]]

    N1=20
    N2=20
    N3=20
    POPS=[N1,N2,N3]
    samples=[msprime.Sample(0,0)]*N1 + [msprime.Sample(1,0)]*N2 + [msprime.Sample(2,0)] *N3


    parametersfile.write('\t'.join([str(x) for x in [COLONIZER,N_locals,N_metropolis,N_initial_colony,N_finale_colony,r_locals,r_metropolis,r_colony]]))
    parametersfile.write('\n')
    parametersfile.write('\t'.join([str(x) for x in migration_matrix]))
    
    
    
    No=1000000
    Theta=No*1e-8*4
    ##metatroph parameters se ms parameters
    
    os.system('ms 60 100 -t {} -I 3 20 20 20 -g 1 {} -g 2 {} -g 3 {} -n 1 {} -n 2 {} -n 3 {} -m 1 2 {} -m 2 1 {} -m 2 3 {} -m 3 2 {} -m 3 1 {} -m 1 3 {}  -ej 0.2 3 {} > MS_OUTPUT_{}'.format(Theta,r_locals,r_metropolis,r_colony,N_locals,N_metropolis,N_finale_colony,migration_matrix[0][1],migration_matrix[0][2],migration_matrix[1][0],migration_matrix[1][2],migration_matrix[2][0],migration_matrix[2][1],COLONIZER,REPS)) 
    os.system('ms 60 100 -t {} -I 3 20 20 20 -g 1 {} -g 2 {} -g 3 {} -n 1 {} -n 2 {} -n 3 {} -m 1 2 {} -m 2 1 {} -m 2 3 {} -m 3 2 {} -m 3 1 {} -m 1 3 {}  -ej 0.2 3 {} '.format(Theta,r_locals,r_metropolis,r_colony,N_locals,N_metropolis,N_finale_colony,migration_matrix[0][1],migration_matrix[0][2],migration_matrix[1][0],migration_matrix[1][2],migration_matrix[2][0],migration_matrix[2][1],COLONIZER,REPS)) 
    os.system('CoMuStats -input MS_OUTPUT_{} -npop 3 20 20 20 -ms > COMUSTATS_{}'.format(REPS,REPS))
    #mhdenismos growth migration rate meta apo ej?
    
    
    
    
    
    
    
    
    
    
    
    
    
    
