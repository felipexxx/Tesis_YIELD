
import os
import psutil
process = psutil.Process(os.getpid())


import pandas as pd 


from sklearn.model_selection import train_test_split #, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper

#from  RicePaper_RandomForest import RandomForestYield
#from  RicePaper_SupportVectorRegression import SVRegressorYield
#from RicePaper_KNRegression import KNRegressorYield
from RicePaper_ANN import ANNRegressorYield


print(__name__, process )
os.chdir("D:/Felipe/Tesis/Prototipo/Version2")
models_folder="Models"
result_folder="Results"

if __name__ == "__main__":
    
    # LOAD DATA
    folderSource="./data/"
    myFile="RicePaper_ModelFinalFormat.csv"
    
    ONE_VARIETY=False # model the prediction for only the variety with more items in the dataset
    #loading yield data
    finalDataSet= pd.read_csv("%s%s"%(folderSource,myFile),
                             sep=",", 
                             na_filter=False, low_memory=False
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
    
    #finalDataSet.drop('Variety', axis=1, inplace=True) 
    
    #finalDataSet=finalDataSet[["TX_30_FREQ"]]#Testing Support Vector Regressor
#     Index(['P_ACCUM', 'P_10_FREQ', 'TM_AVG', 'TX_AVG', 'TX_30_FREQ', 'RH_AVG',
#       'SR_ACCUM'],
#      dtype='object')                                
      
    #finalDataSet=finalDataSet[["Variety","TX_AVG"]]                                               
    X_train, X_test, y_train, y_test = train_test_split(finalDataSet.values, yieldY, test_size=0.3)
    print( X_train.shape, y_train.shape)
    print (X_test.shape, y_test.shape)
    # END LOAD DATA
    
    
    columnsResult=["RMSE", "RRSE", "R2_TEST", "MAE", "TEST_TIME","TUNIN_TIME","R2_TRAINED", "K_FOLD","JOBS","GRID_CASES","NAME"]
    
    print("In Main")
    
    jobsList=[1,2,3]
    
    modelsList=[ANNRegressorYield(models_folder)]#[RandomForestYield(models_folder)]
    df_values_result=[]
    for model in modelsList:
        for numJobs in jobsList:
            
            print("Tunin "+model.NAME)
            modelML,timeTunin = model.hyper_tunin(X_train,y_train,numJobs)
            
            r2Training=modelML.best_score_
            trainingMetric=[timeTunin, r2Training , 10, numJobs, len(modelML.cv_results_["params"]), model.NAME ]
            
            print("Evaluating "+model.NAME)
            resultMetric=model.evaluate(modelML,X_test, y_test)
            resultMetric.extend(trainingMetric)
            
            df_values_result.append(resultMetric)
            
    print(modelML.cv_results_["mean_fit_time"])
    print(modelML.cv_results_["mean_fit_time"].max())
    print(modelML.cv_results_["mean_fit_time"].min())
    print(modelML.cv_results_["mean_fit_time"].std())
    df=pd.DataFrame(df_values_result, columns=columnsResult)
    df=df.round(2)
    df.to_csv("./%s/results_evaluations.csv"%(result_folder))
    print("End Main")

