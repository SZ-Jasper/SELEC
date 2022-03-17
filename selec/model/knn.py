import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from dataprepare.dataprep import *


def kfold(n_split,rand_state):
    """ This is the k-fold cross validation """
    kf = KFold(n_splits=n_split, random_state=rand_state, shuffle= True)
    return kf


def grid_knn_hp(lower, upper, df, output):
    """ This function tunes the hyperparameters """
    param_grid = {'n_neighbors':range(lower,upper),
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree', 'kd_tree'],
                 }
    grid_search = GridSearchCV(KNeighborsRegressor(), 
                               param_grid,
                               cv=5
                              )
    train,test,X_train,y_train,X_test,y_test = data_split(df, 0.2, output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    grid_search.fit(X_train_scaled, y_train)
    best_knn_hp = list(grid_search.best_params_.values())
    alg = best_knn_hp[0]
    n_neigh = best_knn_hp[1]
    weight = best_knn_hp[2]
    return  alg, n_neigh, weight


def knn_train(df, output):
    """ This function train the model and calculate the train error """
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    np.random.seed(66)
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model_train =KNeighborsRegressor( algorithm = alg, 
                                         n_neighbors=n_neigh, 
                                         weights = weight)
    train_results =[]
    train_results_name =['Experimental','Predicted ','RMSE']
    kf = kfold(10,66)
    train,test,X_train,y_train,X_test,y_test = data_split(df, 0.2, 
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
    train,test,X_train,y_train,X_test,y_test = data_split(df, 0.2, 
                                                          output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    KNN_model.fit(X_train_scaled,y_train)
    y_predict=KNN_model.predict(X_test_scaled)
    mse = mean_squared_error(y_predict, y_test)
    RMSE = np.sqrt(mse)
    return RMSE