# -*- coding: utf-8 -*-
print("Pasa por la clase RF")

import numpy as np


#performance 
import time
import os
import psutil
process = psutil.Process(os.getpid())

#Modelling

from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

class ANNRegressorYield:
    
    
    NAME="MLPRegressor"
    
    def __init__(self, models_folder):
        self.models_folder=models_folder
        pass
        
    
    def hyper_tunin(self, X_train, y_train , num_jobs):                                   
        # RANDOM FOREST
        print("Neural")
        
        #Dummy Model  
        self.dummyModel=DummyRegressor()
        self.dummyModel.fit(X_train,y_train)
        
        #save the scalers
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        
        X_train= self.sc_X.fit_transform(X_train)
        y_train= self.sc_y.fit_transform(y_train.values.reshape(-1,1)).reshape(1,-1)[0]
    
    
#        param_grid = {# Simple Grid 
#            'bootstrap': [True],
#            'max_depth': [80],
#            'max_features': [2,3],
#            'min_samples_leaf': [5],
#            'min_samples_split': [8],
#            'n_estimators': [500,1000]
#        }
        
        param_grid = {
            'hidden_layer_sizes': [(2,),(2,2),(5,), (5,5,)],
            #'activation':['identity', 'logistic', 'tanh', 'relu'],
            'activation':[ 'logistic', 'tanh', 'relu'],
            'learning_rate_init':[0.001, 0.3],
            'solver':['lbfgs',  'sgd', 'adam'],
            'momentum':[0.01, 0.4, 0.9],
            'random_state':[42]           
        }
         
#        param_grid = {
#            'n_neighbors':[2,5,10],
#        }
        
        
        #multiprocessing.set_start_method('forkserver') #is already spawn method
        regressorGrid=MLPRegressor()#old   
        grid= GridSearchCV(estimator=regressorGrid, param_grid=param_grid, cv=10, n_jobs=num_jobs, return_train_score=True)
        time_start = time.clock()
        grid= grid.fit(X_train, y_train)
        time_elapsed = (time.clock() - time_start)
                             
   
        joblib.dump([grid,self.dummyModel ], "./%s/%s"%(self.models_folder,self.NAME))#Persistence for the model  
        
        return grid,time_elapsed

    def evaluate(self, model, X_test, y_test):
        
        time_start = time.clock()
        
        X_transformed=self.sc_X.transform(X_test)
        y_transformed=self.sc_y.fit_transform(y_test.values.reshape(-1,1)).reshape(1,-1)[0]
        
        
        y_scaled=model.predict(X_transformed)
        
        y_predict = self.sc_y.inverse_transform(y_scaled.reshape(-1,1)).reshape(1,-1)[0]
        
        y_dummie=self.dummyModel.predict(X_test)     
        
        scoreMy2=model.score(X_transformed, y_transformed) 
        
        
        RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_predict))
        MAE=metrics.mean_absolute_error(y_test, y_predict)
        
        
        R2=scoreMy2
        RMSE_DUMMIE= np.sqrt(metrics.mean_squared_error(y_test, y_dummie ))
        RRSE = RMSE/RMSE_DUMMIE
        
        
        time_elapsed = (time.clock() - time_start)
        
        
        return [RMSE, RRSE, R2, MAE, time_elapsed]
    
    def loadTuninPersistence(self):
        return joblib.load("./%s/%s"(self.models_folder,self.NAME))