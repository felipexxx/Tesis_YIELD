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
myFile="RicePaper_ModelFinalFormat.csv"

ONE_VARIETY=False # model the prediction for only the variety with more items in the dataset
#loading yield data
finalDataSet= pd.read_csv("%s%s"%(folderSource,myFile),
                         sep=",", 
                         na_filter=False, low_memory=False
                         #converters={2:lambda x: pd.to_numeric(x)}#column to function ineficient
                         )



mapper = DataFrameMapper([
    ('Variety', LabelEncoder())
], df_out=True, default=None)

finalDataSet = mapper.fit_transform(finalDataSet.copy())

if (ONE_VARIETY):
    finalDataSet= finalDataSet[finalDataSet["Variety"]==14.0]
    finalDataSet.drop('Variety', axis=1, inplace=True)
    finalDataSet.reset_index(inplace=True, drop=True)


yieldY=finalDataSet["Yield"]
finalDataSet.drop('Yield', axis=1, inplace=True)    
#yieldY=(yieldY-yieldY.min())/yieldY.max()

from scipy.stats import variation                                               
for column in finalDataSet.columns: 
    #finalDataSet[column] = (finalDataSet[column]-finalDataSet[column].mean())/finalDataSet[column].std()
    print(column, variation(finalDataSet[column]))
print(finalDataSet.describe())    
#    plt.figure(figsize=(8,6))
#    plt.plot(finalDataSet[column], yieldY, 'o')
#    plt.xlabel(column)                                       
#    plt.ylabel('YIELD')
#    plt.show()                                        
                                                 
x_train, x_test, y_train, y_test = train_test_split(finalDataSet.values, yieldY, test_size=0.2)
print( x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
#(353, 10) (353,)
#(89, 10) (89,)                                                
#                                                
#                                                
treeClassifier1 = DecisionTreeRegressor(max_depth=7, random_state=8)
treeClassifier2 = DecisionTreeRegressor(max_depth=5)


treeClassifier1.fit(x_train, y_train)
#treeClassifier2.fit(x_train, y_train)

y_pred = treeClassifier1.predict(x_test)  
##apply machine learning function in this place

df2=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  

my_lim=(y_test.min()-1000,y_test.max()+1000)

df2.plot(kind='scatter',x='Actual', y='Predicted',
         xlim=my_lim , ylim = my_lim, title ="Decision Tree Regressor"
         )

scoreMy=treeClassifier1.score(x_test, y_test) 

print('Square:', scoreMy)   
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


# RANDOM FOREST
print("RF")
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=5000, random_state = 20, max_features=2)
regressor.fit(x_train,y_train)


rf_y_pred = regressor.predict(x_test)  
##apply machine learning function in this place

df3=pd.DataFrame({'Actual':y_test, 'Predicted':rf_y_pred})  


df3.plot(kind='scatter',x='Actual', y='Predicted',
         xlim=my_lim , ylim = my_lim, title ="Random Forest Regressor"
         )

scoreMy2=regressor.score(x_test, y_test) 

print('Square:', scoreMy2)   
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rf_y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_y_pred))) 