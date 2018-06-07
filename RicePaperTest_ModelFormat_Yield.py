# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

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
        
        lambda x: (x["P"]>PREP_LIMIT).sum()/(x["P"].count()),#P_10_FREQ
                  
        lambda x: x["TM"].mean(),#TM
        
        lambda x: x["TX"].mean(),#TX
        
        lambda x: (x["TX"]>TEMP_LIMIT).sum()/(x["TX"].count()),#TX_30_FREQ
        
        lambda x: x["RH.x"].mean(),#RH_AVG  
        
        lambda x: x["SR.x"].sum()#SR_ACCUM 
]



#loading yield data
yieldDF= pd.read_csv("%s%s"%(folderSource,myFile),
                         sep=",", 
                         na_filter=False, low_memory=False
                         #converters={2:lambda x: pd.to_numeric(x)}#column to function ineficient
                         )

#yieldDF=yieldDF[yieldDF["Variety"]=="F733"]  
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



finalColumns= np.append(columnsAgg,["Yield","Variety"])
finalDataSet=finalDataSet[finalColumns]

#yieldY=finalDataSet["Yield"]

finalDataSet.to_csv("./data/RicePaper_ModelFinalFormat.csv", index=False)
#
#finalDataSet.drop('Yield', axis=1, inplace=True)
##x_train,y_train=finalDataSet.values, finalDataSet["Yield"].values
##x_test, y_test=finalDataSet.values, finalDataSet["Yield"].values
#
##yieldY=(yieldY-yieldY.mean())/yieldY.std()
#
#
#mapper = DataFrameMapper([
#    ('Variety', LabelEncoder())
#], df_out=True, default=None)
#
#finalDataSet = mapper.fit_transform(finalDataSet.copy())
#
#
##yieldY=(yieldY-yieldY.min())/yieldY.max()
#
#from scipy.stats import variation                                               
#for column in finalDataSet.columns: 
#    #finalDataSet[column] = (finalDataSet[column]-finalDataSet[column].mean())/finalDataSet[column].std()
#    print(column, variation(finalDataSet[column]))
#print(finalDataSet.describe())    
##    plt.figure(figsize=(8,6))
##    plt.plot(finalDataSet[column], yieldY, 'o')
##    plt.xlabel(column)                                       
##    plt.ylabel('YIELD')
##    plt.show()                                        
#                                                 
#x_train, x_test, y_train, y_test = train_test_split(finalDataSet.values, yieldY, test_size=0.2)
#print( x_train.shape, y_train.shape)
#print (x_test.shape, y_test.shape)
##(353, 10) (353,)
##(89, 10) (89,)                                                
##                                                
##                                                
#treeClassifier1 = DecisionTreeRegressor(max_depth=7)
#treeClassifier2 = DecisionTreeRegressor(max_depth=5)
#
#
#
#
#treeClassifier1.fit(x_train, y_train)
##treeClassifier2.fit(x_train, y_train)
#
#y_pred = treeClassifier1.predict(x_test)  
###apply machine learning function in this place
#
#df2=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
#
#df2.plot(kind='scatter',x='Actual', y='Predicted')
#
#scoreMy=treeClassifier1.score(x_test, y_test) 
#
#print('Square:', scoreMy)   
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 



