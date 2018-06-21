# -*- coding: utf-8 -*-
print("Pasa por la clase RF")

import pandas as pd 
import numpy as np


#performance 
import time
import os
import psutil
process = psutil.Process(os.getpid())

#Modelling

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.externals import joblib

class RandomForestYield:
    
    
    NAME="RandomForestRegressor"
    
    def __init__(self, models_folder):
        self.models_folder=models_folder
        pass
        
    
    def hyper_tunin(self, X_train, y_train , num_jobs):                                   
        # RANDOM FOREST
        print("RF")
         
        
#        param_grid = {# Simple Grid 
#            'bootstrap': [True],
#            'max_depth': [80],
#            'max_features': [2,3],
#            'min_samples_leaf': [5],
#            'min_samples_split': [8],
#            'n_estimators': [500,1000]
#        }
        
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [ 300, 1000, 2000]
        }
        
        #multiprocessing.set_start_method('forkserver') #is already spawn method
        regressorGrid=RandomForestRegressor( random_state = 42)#old   
        grid= GridSearchCV(estimator=regressorGrid, param_grid=param_grid, cv=10, n_jobs=num_jobs, return_train_score=True)
        time_start = time.clock()
        grid= grid.fit(X_train, y_train)
        time_elapsed = (time.clock() - time_start)
                             
   
        self.dummyModel=DummyRegressor()
        self.dummyModel.fit(X_train,y_train)
        
        joblib.dump([grid,self.dummyModel ], "./%s/%s"%(self.models_folder,self.NAME))#Persistence for the model  
        
        return grid,time_elapsed

    def evaluate(self, model, X_test, y_test):
        
        time_start = time.clock()
        y_predict = model.predict(X_test)
        
        y_dummie=self.dummyModel.predict(X_test)     
        
        scoreMy2=model.score(X_test, y_test) 
        
        
        RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_predict))
        MAE=metrics.mean_absolute_error(y_test, y_predict)
        R2=scoreMy2
        RMSE_DUMMIE= np.sqrt(metrics.mean_squared_error(y_test, y_dummie ))
        RRSE = RMSE/RMSE_DUMMIE
        
        
        time_elapsed = (time.clock() - time_start)
        
        
        return [RMSE, RRSE, R2, MAE, time_elapsed]
    
    def loadTuninPersistence(self):
        return joblib.load("./%s/%s"(self.models_folder,self.NAME))