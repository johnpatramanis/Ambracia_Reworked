mydir=getwd()
setwd(mydir)

library(dplyr)
library(abcrf)

a <- read.table("FOR_SVM", h=F)

cross_val_data=sample_n(a,1000)
a=anti_join(a,cross_val_data)

#build training dataset
modindex= a[,1]  #indexes of model
modindex=sapply(modindex,as.factor)
sumsta= a[,-(1:14)] #summary stats
data1 = data.frame(modindex, sumsta) #combined


model.rf1 <- abcrf(modindex~., data = data1,ncores=16, ntree=10000,lda=FALSE) #our model, does not run with lda=TRUE


sumstb= cross_val_data[,-(1:14)]
truindex= cross_val_data[,1]
truindex=sapply(truindex,as.factor)

err.rf <- err.abcrf(model.rf1, data1)
print(err.rf)



model.rf.r <- regAbcrf(modindex~., data1,ncores=4, ntree=10000)
print(model.rf.r)



myerror=err.regAbcrf(model.rf.r, data1)
print(myerror)



model_predict=predict(model.rf1,sumstb,data1,ntree=100)
print(model_predict)



model_predict=predict(model.rf1,sumstb,data1,ntree=1000)
print(model_predict)


model_predict=predict(model.rf1,sumstb,data1,ntree=10000)
print(model_predict)

