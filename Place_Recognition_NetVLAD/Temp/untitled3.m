indexvaltrain = table2cell(indexvaltrain);
dbStruct.qImageFns = indextrainquery
%%
A = table2array(utmtrainquery);
dbStruct.utmQ = A'
%%
datetraintrain = table2array(datetraintrain)
dbStruct.qTimeStamp = A'
%%
datetrainquery = table2array(datetrainquery)
dbStruct.qTimeStamp = datetrainquery'
%%
clc