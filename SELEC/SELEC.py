### SELEC functions

### One Hot Encoding dataset
def ohe_ACE(df):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import sklearn
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ACE = ohe.fit_transform(ACE)
    df_encoded_ACE = pd.DataFrame(ACE.toarray())
    return df_encoded_ACE

def ohe_dataframe(df):
    import pandas as pd    
    df_ACE_col_name = ['A1','C1','C2','C3','E1','E2','E3']
    df_encoded_ACE = ohe_ACE(df)
    for i in range(len(df_encoded_ACE.columns)):
        df_encoded_ACE = df_encoded_ACE.rename({df_encoded_ACE.columns[i]: df_ACE_col_name[i]}, axis=1) 
    df_ohe = pd.concat([df_encoded_ACE, df],axis=1)
    return df_ohe

### Split Data
def data_split (df, test_ratio, output, seed):
    import numpy as np
    import pandas as pd
    np.random.seed(seed)
    df_enbattery = ohe_dataframe(df)
    total_row = df_enbattery.shape[0]
    test_row = round(total_row *test_ratio)
    train_row = total_row -test_row
    indices =np.random.permutation(total_row)
    train_indx, test_idx =indices[:train_row], indices[train_row:]
    train,test = df_enbattery.iloc[train_indx,:], df_enbattery.iloc[test_idx,:]
    X_test = test[['A1','C1','C2','C3','E1','E2','E3','Cycle','temperature','discharge_crate']]
    y_test = test[[output]]
    X_train = train[['A1','C1','C2','C3','E1','E2','E3','Cycle','temperature','discharge_crate']]
    y_train = train[[output]]
    return train,test,X_train,y_train,X_test,y_test

### Kfolds
def kfold(n_split,rand_state):
    import sklearn
    from sklearn.model_selection import KFold 
    kf = KFold(n_splits=n_split, random_state=rand_state, shuffle= True)
    return kf

### Scale train and test sets
def data_scale(X_train, X_test):
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import StandardScaler,scale,MinMaxScaler
    scaler = StandardScaler(with_mean=True,with_std=True)
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    scaler.fit(X_test)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    return X_train_scaled,X_test_scaled

### GridSearchCV for Hyperparameters
def grid_knn_hp(lower, upper, df, output):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
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

### Train machine learning models
def knn_train(df, output):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    np.random.seed(66)
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model_train =KNeighborsRegressor( algorithm = alg, n_neighbors=n_neigh, weights = weight)
    train_results =[]
    train_results_name =['Experimental','Predicted ','RMSE']
    kf = kfold(10,66)
    train,test,X_train,y_train,X_test,y_test = data_split(df, 0.2, output, 66)
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

### Test Machine learning models
def knn_test(df, output):
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    np.random.seed(66)
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model = KNeighborsRegressor( algorithm = alg, n_neighbors=n_neigh, weights = weight)
    train,test,X_train,y_train,X_test,y_test = data_split(df, 0.2, output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    KNN_model.fit(X_train_scaled,y_train)
    y_predict=KNN_model.predict(X_test_scaled)
    mse = mean_squared_error(y_predict, y_test)
    RMSE = np.sqrt(mse)
    return RMSE

### Prepare battery sets for predictor
def X_set_in(df):
    import numpy as np
    import pandas as pd
    ac = df['anode'].unique()
    cc = df['cathode'].unique()
    ec = df['electrolyte'].unique()
    cycle = df['Cycle'].unique()
    tc = df['temperature'].unique()
    cr = df['discharge_crate'].unique()
    X_temp = []
    for a in range(len(ac)):
        for b in range(len(cc)):
            for c in range(len(ec)):
                for d in range(len(cycle)):
                    for e in range(len(tc)):
                        for f in range(len(cr)):
                            X_temp.append([ac[a],cc[b],ec[c],cycle[d],tc[e],cr[f]])
    column_names = ['anode','cathode','electrolyte','Cycle','temperature','discharge_crate']
    X_set = pd.DataFrame(X_temp, columns = column_names)
    return X_set

def X_set_en(df):
    import numpy as np
    import pandas as pd
    import pickle
    import sklearn
    from sklearn.preprocessing import OneHotEncoder
    X_set = X_set_in(df)
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ohe.fit_transform(ACE)
    filehandler = open("ohe.obj","wb")
    pickle.dump(ohe,filehandler)
    filehandler.close()
    file = open("ohe.obj",'rb')
    ohe_loaded = pickle.load(file)
    file.close()
    ace = X_set.loc[:,['anode','cathode','electrolyte']]
    ace = ohe_loaded.transform(ace)
    ace = pd.DataFrame(ace.toarray())
    X_num = X_set.loc[:,['Cycle','temperature','discharge_crate']]
    df_ace_col_name = ['A1','C1','C2','C3','E1','E2','E3']
    for i in range(len(ace.columns)):
        ace = ace.rename({ace.columns[i]: df_ace_col_name[i]}, axis=1) 
    X_set = pd.concat([ace, X_num],axis=1)
    return X_set

### Predictor
def df_prep(df, output):    
    import pandas as pd
    df_encoded_ACE = ohe_ACE(df)
    df_enbattery = ohe_dataframe(df)
    X_bat = df_enbattery.loc[:,['A1','C1','C2','C3','E1','E2','E3','Cycle','temperature','discharge_crate']]
    y_bat = df_enbattery.loc[:,[output]]
    return X_bat, y_bat

def battery_predictor(df, output):    
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.neighbors import KNeighborsRegressor
    X_bat, y_bat = df_prep(df, output)
    X_set = X_set_en(df)
    X_bat_scaled, X_set_scaled = data_scale(X_bat, X_set)
    best_knn_hp = grid_knn_hp(1, 51, df, output)
    alg = best_knn_hp[0]
    n_neigh = best_knn_hp[1]
    weight = best_knn_hp[2]
    np.random.seed(66)
    KNN_model =KNeighborsRegressor(algorithm=alg, n_neighbors=n_neigh, weights=weight)
    KNN_model.fit(X_bat_scaled,y_bat)
    y_predict=KNN_model.predict(X_set_scaled)
    return y_predict

### Report Generator
def report_gen(df):
    import numpy as np
    import pandas as pd
    in_set = X_set_in(df)
    CC = pd.DataFrame(battery_predictor(df, 'Charge_Capacity (Ah)'), columns = ['Charge_Capacity (Ah)'])
    DC = pd.DataFrame(battery_predictor(df, 'Discharge_Capacity (Ah)'), columns = ['Discharge_Capacity (Ah)'])
    CE = pd.DataFrame(battery_predictor(df, 'Charge_Energy (Wh)'), columns = ['Charge_Energy (Wh)'])
    DE = pd.DataFrame(battery_predictor(df, 'Discharge_Energy (Wh)'), columns = ['Discharge_Energy (Wh)'])
    CEff = pd.DataFrame(battery_predictor(df, 'Coulombic_Efficiency (%)'), columns = ['Coulombic_Efficiency (%)'])
    EEff = pd.DataFrame(battery_predictor(df, 'Energy_Efficiency (%)'), columns = ['Energy_Efficiency (%)'])
    report = pd.concat([in_set,CC,DC,CE,DE,CEff,EEff],axis=1)
    return report

