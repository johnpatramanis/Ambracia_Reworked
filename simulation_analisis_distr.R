
setwd("/home/kluser2/datasets/ambracia_sims/REWORKED/")




ComusMinMax <- read.table("COMUS_MIN_MAX",sep='\t')


for (i in 0:29000){
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
par_min=ComusMinMax[k,1]
par_max=ComusMinMax[k,2]
  
newcomusdistributions[j]=list(density(mycomusdataframe,n=11,from=par_min,to=par_max)$y)
j=j+1
  
  
}



for (w in 1:length(ParametersList[[1]])){

cat(paste(ParametersList[[1]][w],'\t'),file='FOR_ABC',append=TRUE,sep='\t')
}
for (w in 1:length(migrations[[1]])){
  #for (c in 1:length(ParametersList[[2]][w])){
    
    cat(paste(migrations[[1]][w],'\t'),file='FOR_ABC',append=TRUE,sep='\t')
    
  #}
}


for (c in 1:length(newcomusdistributions)){
  
  for (w in 1:length(newcomusdistributions[[c]])){
    
  cat(paste(newcomusdistributions[[c]][w],'\t'),file='FOR_ABC',append=TRUE,sep='\t')  
	}
  
  
}
cat('',file='FOR_ABC',append=TRUE,sep='\n')

}


system('grep -vwE "NA" FOR_ABC > FOR_ABC_CLEAN')

#a <- read.table("FOR_ABC_CLEAN", h=F)

#dim(a)# check dims


