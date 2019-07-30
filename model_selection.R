mydir=getwd()
setwd(mydir)

library(dplyr)
library(abcrf)

a <- read.table("FOR_ABC_CLEAN", h=F)

cross_val_data=sample_n(a,1000)
a=anti_join(a,cross_val_data)


modindex= a[,1]  #indexes of model
sumsta= a[,-(1:14)] #summary stats
data1 = data.frame(modindex, sumsta) #combined?


model.rf1 <- abcrf(modindex~., data = data1,ncores=8, ntree=100,lda=FALSE) #our model, does not run with lda=TRUE