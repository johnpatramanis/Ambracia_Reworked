mydirectory=getwd()
setwd(mydirectory)

library(abc)
library(Metrics)
library(dplyr)
k=5
library(RColorBrewer); my.cols <- rev(brewer.pal(k, "RdYlBu")) 


a <- read.table("FOR_SVM", h=F)
dim(a)

LOGIT_MATRIX=matrix(nrow=13,ncol=2)
for (Z in 1:13){
  
  LOGIT_MATRIX[Z,1]=min(a[,Z+1])
  LOGIT_MATRIX[Z,2]=max(a[,Z+1])
  
}


cross_val_data=sample_n(a,1000)
a=anti_join(a,cross_val_data)


params <- a[,2:14]   
stats <- a[,-(1:14)]
test <- cross_val_data[1,-(1:14)]
test_params <-cross_val_data[1,2:14]



myabc <- abc(target=test, param=params, sumstat=stats, tol=0.1, method="ridge", hcorr=TRUE,transf=c('logit'),logit.bounds=LOGIT_MATRIX)
summarystats=summary(myabc)

#plot examples

priors=cbind(params[,1],stats[,1])
posteriors=cbind(myabc$adj.values[,1],myabc$ss[,1])

