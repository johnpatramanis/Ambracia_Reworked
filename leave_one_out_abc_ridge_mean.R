
mydirectory=getwd()
setwd(mydirectory)



system('grep -vwE "NA" FOR_ABC > FOR_ABC_CLEAN')
library(abc)
library(Metrics)
library(dplyr)

a <- read.table("FOR_ABC_CLEAN", h=F)
dim(a)
LOGIT_MATRIX=matrix(nrow=13,ncol=2)
for (Z in 1:13){

LOGIT_MATRIX[Z,1]=min(a[,Z+1])
LOGIT_MATRIX[Z,2]=max(a[,Z+1])
    
}
print(LOGIT_MATRIX)

predicted=vector()
actual=vector()
mean_diff=vector()

cross_val_data=subset(a,1000)
a=anti_join(a,cross_val_data)




for (j in 1:dim(cross_val_data)[1]){
  
tryCatch({  

params <- a[,2:14]   #leave one out
stats <- a[,-(1:14)] # << ,<<
test <- cross_val_data[j,-(1:14)]
test_params <-cross_val_data[j,2:14]

dim(params)
dim(stats) # check dims to make sure


  
myabc <- abc(target=test, param=params, sumstat=stats, tol=0.1, method="ridge", hcorr=TRUE,transf=c('logit'),logit.bounds=LOGIT_MATRIX)

summarystats=summary(myabc)


for (w in 1:13){

PredictionsFile <- paste ("PREDICTIONS_MEAN_",w, sep = "", collapse = NULL)

print(j)
print(summarystats)
print(summarystats[3,w])
predicted1=as.numeric(summarystats[3,w])# MED
predicted2=as.numeric(summarystats[4,w])# MEAN
predicted3=as.numeric(summarystats[5,w])# MODE
actual=as.numeric(test_params[w])
print(predicted1)
print(predicted2)
print(predicted3)
print(actual)



cat(paste(actual,'\t',predicted1,'\t',predicted2,'\t',predicted3,'\n'),file=PredictionsFile,append=TRUE,sep='\t')

}
}, error=function(e){})

}

print(mae(actual,predicted))


