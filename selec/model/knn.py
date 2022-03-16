import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,scale,MinMaxScaler
from sklearn.metrics import mean_squared_error

from batt_descriptor.batt_describe import ohe_dataframe

df_battery = pd.read_csv('../data/Battery_Dataset.csv') 


def data_split (df, test_ratio, output, seed):
    """ This function split the data"""
    np.random.seed(seed)
    
    df_enbattery = ohe_dataframe(df)

    total_row = df_enbattery.shape[0]
    test_row = round(total_row *test_ratio)
    train_row = total_row -test_row
    
    indices =np.random.permutation(total_row)
    train_indx, test_idx =indices[:train_row], indices[train_row:]
    train,test = df_enbattery.iloc[train_indx,:], df_enbattery.iloc[test_idx,:]
    
    X_test = test[['A1','C1','C2','C3','E1','E2','E3',
                   'Cycle','temperature','discharge_crate']]
    y_test = test[[output]]
    
    X_train = train[['A1','C1','C2','C3','E1','E2','E3',
                     'Cycle','temperature','discharge_crate']]
    y_train = train[[output]]
    
    return train,test,X_train,y_train,X_test,y_test


def kfold(n_split,rand_state):
    """ This is the k-fold cross validation """
    kf = KFold(n_splits=n_split, random_state=rand_state, shuffle= True)
    return kf


def data_scale(X_train, X_test):
    """ This funciton insure all the input values in a standar range """
    scaler = StandardScaler(with_mean=True,with_std=True)

    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    
    scaler.fit(X_test)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    
    return X_train_scaled,X_test_scaled


def grid_knn_hp(lower, upper, df_battery, output):
    """ This function tuning the hyperparameters """
    param_grid = {'n_neighbors':range(lower,upper),
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree', 'kd_tree'],
                 }


    grid_search = GridSearchCV(KNeighborsRegressor(), 
                               param_grid,
                               cv=5
                              )
    
    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, 
                                                          output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn_hp = list(grid_search.best_params_.values())
    alg = best_knn_hp[0]
    n_neigh = best_knn_hp[1]
    weight = best_knn_hp[2]
    
    return  alg, n_neigh, weight


def knn_train(df, output):
    """ This function train the model and calculate the train error """
    np.random.seed(66)
    
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model_train =KNeighborsRegressor( algorithm = alg, 
                                         n_neighbors=n_neigh, 
train_results =[]
train_results_name =['Experimental','Predicted ','RMSE']                                         weights = weight)

    kf = kfold(10,66)

    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, 
                                                          output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    
    for train_index, test_index in kf.split(X_train_scaled):
        X_training, X_validate = X_train_scaled.iloc[train_index], X_train_scaled.iloc[test_index]
        y_training, y_validate = y_train.iloc[train_index], y_train.iloc[test_index]

        np.random.seed(66)
        KNN_model_train.fit(X_training,y_training)
        y_train_predicted = KNN_model_train.predict(X_validate)

        mse = mean_squared_error(y_train_predicted, y_validate)
        RMSE = np.sqrt(mse)

        train_results.append([y_validate,y_train_predicted,RMSE])
    
    Train_results = pd.DataFrame (train_results,columns=train_results_name)
    RMSE_avg = np.average(Train_results['RMSE'])
    
    return RMSE_avg


def knn_test(df, output):
    """ This function predict the result and calculate the test error """
    np.random.seed(66)
    
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model = KNeighborsRegressor( algorithm = alg, 
                                    n_neighbors=n_neigh, 
                                    weights = weight)
    
    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 
                                                          0.2, 
                                                          output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    
    KNN_model.fit(X_train_scaled,y_train)
    y_predict=KNN_model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_predict, y_test)
    RMSE = np.sqrt(mse)
    return RMSE