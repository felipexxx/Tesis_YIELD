# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np


#directory
folderSource="./data/"
myFile="cropYieldCleaned.csv"
myFileClimate="Saldana.txt"

operators=["sum","mean","sum","max","min"]
variables=["RAIN","RHUM", "SR", "TMAX",  "TMIN"]

TEMP_LIMIT=30.0
PREP_LIMIT=10.0

functionsStats={"sum":np.sum,"max":np.max,"min":np.min,"mean":np.mean}

# variables aggregation from weather data 
columnsAgg=["P_ACCUM","P_10_FREQ","TM_AVG", "TX_AVG","TX_30_FREQ", "RH_AVG","SR_ACCUM"]
functionsAgg=[
        lambda x: x["P"].sum(),#P_ACCUM
        
        lambda x: (x["P"]>PREP_LIMIT).sum(),#P_10_FREQ
                  
        lambda x: x["TM"].mean(),#TM
        
        lambda x: x["TX"].mean(),#TM
        
        lambda x: (x["TX"]>TEMP_LIMIT).sum(),#TX_30_FREQ
        
        lambda x: x["RH.x"].sum(),#RH_AVG  
        
        lambda x: x["SR.x"].sum()#SR_ACCUM 
]



#loading yield data
yieldDF= pd.read_csv("%s%s"%(folderSource,myFile),
                         sep=",", 
                         na_filter=False, low_memory=False
                         #converters={2:lambda x: pd.to_numeric(x)}#column to function ineficient
                         )  
#loading weather data
climateDF= pd.read_csv("%s%s"%(folderSource,myFileClimate),
                         sep=" ", 
                         na_filter=False, low_memory=False, index_col=0
                         #converters={2:lambda x: pd.to_numeric(x)}#column to function ineficient
                         )  
dateFormat="%Y-%m-%d"

#format date fields from string to date type
yieldDF['Sowing_Date']=pd.to_datetime(yieldDF['Sowing_Date'], format=dateFormat)
yieldDF['Harvest_Date']=pd.to_datetime(yieldDF['Harvest_Date'], format=dateFormat)
climateDF.index=pd.to_datetime(climateDF.index, format=dateFormat)

allResult=[]

#loop for apply all functions defined in functionsAgg for every trial in yieldData
for index, row in yieldDF.iterrows():
    sowing=row["Sowing_Date"]
    harvest=row["Harvest_Date"]
    mask = (climateDF.index >= sowing) & (climateDF.index <= harvest)
    tempClimate= climateDF[mask]
    
    
    resultAgg=[]
    
    for myFunc in functionsAgg:
        result=myFunc(tempClimate)
        resultAgg.append(result)
   
   
    allResult.append(resultAgg)
    
    
# from matrix allresults to dataframe   
df=pd.DataFrame(allResult,columns=columnsAgg , index=yieldDF.index )      

#concat new columns to yield
finalDataSet=pd.concat([yieldDF,df], axis=1)



finalColumns= np.append(columnsAgg,["Variety","Yield"])
finalDataSet=finalDataSet[finalColumns]

##apply machine learning function in this place




  




