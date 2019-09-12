mydirectory=getwd()
setwd(mydirectory)






for (i in 0:290000){
print(i)
############################################################
#PARAMETERS input  

ParametersFile <- paste ("PARAMETERS_",i, sep = "", collapse = NULL)
con  <- file(ParametersFile, open = "r")

ParametersList <- list()
j=1
while ( TRUE ) {
  line = readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  
  ParametersList[j] <- strsplit(line,'\t')
  j <- j + 1
}

close(con)  

  
migrations = list(unlist(stringr::str_extract_all(ParametersList[[2]],'([0-9]+.[0-9]+)'),recursive = TRUE))
  
  
  
  
############################################################  
#COMUStats input

ComusFile <- paste ("COMUSTATS_",i, sep = "", collapse = NULL)
con  <- file(ComusFile, open = "r")

ComusList <- list()
j=1
while ( TRUE ) {
  line = readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  
  ComusList[j] <- strsplit(line,'\t')
  j <- j + 1
}

close(con)  

comusdataframe=data.frame(matrix(as.numeric(unlist(ComusList[-1])),nrow=length(ComusList)-1,byrow=T))

j=1
newcomusdistributions=list()
for (k in 1:ncol(comusdataframe)){
  
mycomusdataframe=na.omit(comusdataframe[,k])

newcomusdistributions[j]=mean(mycomusdataframe)
j=j+1
  
  
}



for (w in 1:length(ParametersList[[1]])){

cat(paste(ParametersList[[1]][w],'\t'),file='FOR_SVM',append=TRUE,sep='\t')
}
for (w in 1:length(migrations[[1]])){
  #for (c in 1:length(ParametersList[[2]][w])){
    
    cat(paste(migrations[[1]][w],'\t'),file='FOR_SVM',append=TRUE,sep='\t')
    
  #}
}


for (c in 1:length(newcomusdistributions)){
  
  for (w in 1:length(newcomusdistributions[[c]])){
    
  cat(paste(newcomusdistributions[[c]][w],'\t'),file='FOR_SVM',append=TRUE,sep='\t')  
	}
  
  
}
cat('',file='FOR_SVM',append=TRUE,sep='\n')

}


system('grep -vwE "NA" FOR_SVM > FOR_SVM_CLEAN')

#a <- read.table("FOR_ABC_CLEAN", h=F)

#dim(a)# check dims
