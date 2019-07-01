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


reps=10000
for REPS in range(0,reps):

    
##############################################################################################################################################
#Simulation Parameters
    
    parametersfile=open('PARAMETERS_{}'.format(REPS),'w')
    
    N_locals=int(round(random.uniform(400.0,1000.0)))
    N_metropolis=int(round(random.uniform(400.0,1000.0)))
    
    generation_time = 20
    T_COLONIZATION=700/generation_time
    
    
    COLONIZER=random.randint(0,1)
    if COLONIZER==0:
        N_initial_colony=int(round(random.uniform(200.0,float(N_locals))))
        while N_initial_colony>N_metropolis:
            N_initial_colony=int(round(random.uniform(200.0,float(N_metropolis))))
    if COLONIZER==1:
        N_initial_colony=int(round(random.uniform(200.0,float(N_metropolis))))



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
    ###############################################################################################################################
    


    population_configurations = [
        msprime.PopulationConfiguration(initial_size=N_locals,growth_rate=r_locals),
        msprime.PopulationConfiguration(initial_size=N_metropolis, growth_rate=r_metropolis),
        msprime.PopulationConfiguration(initial_size=N_finale_colony, growth_rate=r_colony)
    ]



    migration_matrix = [
        [0,10**(-1*random.uniform(1,4)),10**(-1*random.uniform(1,4))],
        [10**(-1*random.uniform(1,4)),0,10**(-1*random.uniform(1,4))],
        [10**(-1*random.uniform(1,4)),10**(-1*random.uniform(1,4)),0]]

    N1=20
    N2=20
    N3=20
    POPS=[N1,N2,N3]
    samples=[msprime.Sample(0,0)]*N1 + [msprime.Sample(1,0)]*N2 + [msprime.Sample(2,0)] *N3

    demographic_events = [
    msprime.MigrationRateChange(time=T_COLONIZATION, rate=0, matrix_index=(0, 2)),
    msprime.MigrationRateChange(time=T_COLONIZATION, rate=0, matrix_index=(2, 0)),
    msprime.MigrationRateChange(time=T_COLONIZATION, rate=0, matrix_index=(1, 2)),
    msprime.MigrationRateChange(time=T_COLONIZATION, rate=0, matrix_index=(2, 1)),
    msprime.PopulationParametersChange(time=T_COLONIZATION, initial_size=N_initial_colony, growth_rate=0, population_id=2),
    msprime.MassMigration(time=T_COLONIZATION, source=2, destination=COLONIZER, proportion=1.0),
    
    
    ]

    parametersfile.write('\t'.join([str(x) for x in [COLONIZER,N_locals,N_metropolis,N_initial_colony,N_finale_colony,r_locals,r_metropolis,r_colony]]))
    parametersfile.write('\n')
    parametersfile.write('\t'.join([str(x) for x in migration_matrix]))
    
    print(migration_matrix)
    
    




######################################################################################################################################################
#RUN the simulation and output genotypes in vcfs and ms format files, one for each chrom 

    
    def SIMULATE(L,argument,samples,population_configurations,migration_matrix,demographic_events):
        j=int(argument)
        #recomb_map=msprime.RecombinationMap.read_hapmap('genetic_map_GRCh37_chr{}.txt'.format(j))
        dd = msprime.simulate(samples=samples,
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,mutation_rate=1e-8,
            demographic_events=demographic_events,length=500000)
        outfile=open('ms_prime_{}'.format(j),'w')
        for var in dd.variants():
            L.append([int(j),var.index,var.position])
            for genotype in var.genotypes:
                outfile.write(str(genotype))
            outfile.write('\n')
        outfile.close()    
        return j,L
    
    
    
    L=[]
    if __name__ == '__main__':
        with Manager() as manager:
            L=manager.list(L)
            processes=[]
####################################################################################################################################################################################
            KOMMATIA=50
            for loop in range(1,KOMMATIA):
                p=Process(target=SIMULATE,args=(L,loop,samples,population_configurations,migration_matrix,demographic_events,))
                processes.append(p)
                
                p.start()
        
                
            for p in processes:
                p.join()
            #print(len(L),'1')
            sys.stdout.flush()
            variants=sorted(list(L))


    variantinfo=['{}\t{}\t{}\n'.format(x[0],x[1],x[2])for x in variants]

    variantinformation=open('variants_info.txt','w')
    variantinformation.write('CHROM\tVARIANT\tPOSITION\n')
    for loop in variantinfo:
        variantinformation.write(loop)
    
    variantinformation.close()

    elapsed_time_1 = time.time() - start_time        
        
    print('Step 1 : {} '.format(elapsed_time_1/60))        
    print(REPS)
        
        
        
        
######################################################################################################################################################
#Transform msprime format files to ms format
#prepare for COMUS stats

    MYRUN=KOMMATIA-1
    MAXRUNS=MYRUN
    MYRUN=1
    while MYRUN<=MAXRUNS:
        
        msfile=open('ms_{}'.format(MYRUN),'w')
        column=0
        while column<len(samples):
            msprimefile=open('ms_prime_{}'.format(MYRUN),'r')
            person=[]
            for line in msprimefile:
                line=line.strip().split()[0]
                person.append(str(line[column]))
            msfile.write(''.join(person))
            msfile.write('\n')
            column+=1
            msprimefile.close()
        MYRUN+=1

######################################################

#MERGING OF FILES ####


    MERGED=open('ms_allchroms_{}'.format(REPS),'w')
    for sample in range(0,len(samples)):
        for chromosome in range(1,KOMMATIA):
            file=open('ms_{}'.format(chromosome),'r')
            myline=0
            for line in file:
                #print(sample,myline)
                if myline==sample:
                    line=line.strip()
                    MERGED.write(str(line))
                    break
                myline+=1
            file.close()
        MERGED.write('\n')
    MERGED.close()
    
## os.system('rm ms_*.')
    
#####################################################
#Split each ms format chromosome file to 50kb chunks

    SNPS=open('variants_info.txt','r')
    firstLine = SNPS.readline()
    CHUNKS=[]
    POSITIONS=[]
    begin=0
    end=0
    counter=0
    chr=1
    
    for line in SNPS:
        line=line.strip().split()
        POSITIONS.append(line[2])
        end=float(line[2])
        if chr!=int(line[0]):
            CHUNKS.append([counter,begin,end])
            begin=0
            chr=int(line[0])
            
        counter+=1
    SNPS.close()
    
    MS_MERGED=open('ms_allchroms_{}'.format(REPS),'r')    
###########PRINT CHUNKS MS FORMAT FOR COMUSTATS
    MS_ALL_CHROMS=[]
    for line in MS_MERGED:
        line=line.strip().split()
        MS_ALL_CHROMS.append(line[0])
    counter=0
    begin=0
    opener=open('CHUNKED_{}'.format(REPS),'w')
    opener.write('ms {} {}\n{} {} {}'.format(len(samples),len(CHUNKS),random.randint(0,10000),random.randint(0,10000),random.randint(0,10000)))
    opener.write('\n')
    for x in CHUNKS:
        opener.write("\n")
        opener.write("//\n")
        opener.write("segsites: {}\n".format(x[0]-begin))
        positions_of_this_chunk=POSITIONS[begin:x[0]]
        positions_of_this_chunk=' '.join(positions_of_this_chunk)
        opener.write("positions: {}".format(positions_of_this_chunk))
        opener.write("\n")

        for y in MS_ALL_CHROMS:                    
            opener.write(''.join(y[begin:x[0]]))
            opener.write('\n')            
        begin=x[0]+1
        
        counter+=1

    elapsed_time_2=time.time() - start_time
    print('step 2 : {}'.format(elapsed_time_2/60))

     
################################################################ RUN COMUSTATS #################################################################

    os.system('CoMuStats -input CHUNKED_{} -npop 3 20 20 20 -ms > COMUSTATS_{}'.format(REPS,REPS))
    elapsed_time_4=time.time() - start_time
    print('step 4 : {}'.format(elapsed_time_4/60)) 
        
###############################################################################################################################################
    os.system('rm CHUNKED*')
    os.system('rm *allchroms*')
    

###############################################################################################################################################


