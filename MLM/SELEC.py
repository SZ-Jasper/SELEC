### One Hot Encoding dataset
def ohe_ACE(df):
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ACE = ohe.fit_transform(ACE)
    df_encoded_ACE = pd.DataFrame(ACE.toarray())
    return df_encoded_ACE

### Split Data
def ohe_dataframe(df):
    df_ACE_col_name = ['A1','C1','C2','C3','E1','E2','E3']
    df_encoded_ACE = ohe_ACE(df)
    for i in range(len(df_encoded_ACE.columns)):
        df_encoded_ACE = df_encoded_ACE.rename({df_encoded_ACE.columns[i]: df_ACE_col_name[i]}, axis=1) 
    df_ohe = pd.concat([df_encoded_ACE, df],axis=1)
    return df_ohe

def data_split (df, test_ratio, output, seed):
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
    kf = KFold(n_splits=n_split, random_state=rand_state, shuffle= True)
    return kf

### Scale train and test sets
def data_scale(X_train, X_test):
        
    scaler = StandardScaler(with_mean=True,with_std=True)

    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    
    scaler.fit(X_test)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    
    return X_train_scaled,X_test_scaled

### GridSearchCV for Hyperparameters
def grid_knn_hp(lower, upper, df, output):
    param_grid = {'n_neighbors':range(lower,upper),
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree', 'kd_tree'],
                 }


    grid_search = GridSearchCV(KNeighborsRegressor(), 
                               param_grid,
                               cv=5
                              )
    
    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn_hp = list(grid_search.best_params_.values())
    alg = best_knn_hp[0]
    n_neigh = best_knn_hp[1]
    weight = best_knn_hp[2]
    
    return  alg, n_neigh, weight

### Train machine learning models
def knn_train(df, output):
    np.random.seed(66)
    
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model_train =KNeighborsRegressor( algorithm = alg, n_neighbors=n_neigh, weights = weight)

    train_results =[]
    train_results_name =['Experimental','Predicted ','RMSE']

    kf = kfold(10,66)

    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, output, 66)
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
    np.random.seed(66)
    
    alg, n_neigh, weight = grid_knn_hp(1, 51, df, output)
    KNN_model = KNeighborsRegressor( algorithm = alg, n_neighbors=n_neigh, weights = weight)
    
    train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, output, 66)
    X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
    
    KNN_model.fit(X_train_scaled,y_train)
    y_predict=KNN_model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_predict, y_test)
    RMSE = np.sqrt(mse)
    return RMSE

### Prepare user battery set for predictor
def X_set_prep(df, desc):
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ohe.fit_transform(ACE)
    
    filehandler = open("ohe.obj","wb")
    pickle.dump(ohe,filehandler)
    filehandler.close()

    file = open("ohe.obj",'rb')
    ohe_loaded = pickle.load(file)
    file.close()

    ec = df_battery['electrolyte'].unique()
    column_names = ['anode','cathode','electrolyte','Cycle','temperature','discharge_crate']
    X_set = pd.DataFrame(columns = column_names)
    for i in range(len(ec)):
        X_set.loc[i] = [desc[0],desc[1],ec[i],desc[2],desc[3],desc[4]]
    
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
    df_encoded_ACE = ohe_ACE(df)
    df_enbattery = ohe_dataframe(df)
    
    X_bat = df_enbattery.loc[:,['A1','C1','C2','C3','E1','E2','E3','Cycle','temperature','discharge_crate']]
    y_bat = df_enbattery.loc[:,[output]]
    
    return X_bat, y_bat

def battery_predictor(df, output, desc):
    X_bat, y_bat = df_prep(df, output)
    X_set = X_set_prep(df, desc)
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
def report_gen(df, desc):
    ec = df_battery['electrolyte'].unique()
    column_names = ['anode','cathode','electrolyte','Cycle','temperature','discharge_crate']
    user_set = pd.DataFrame(columns = column_names)
    for i in range(len(ec)):
        user_set.loc[i] = [description[0],description[1],ec[i],description[2],description[3],description[4]]
    
    CC = pd.DataFrame(battery_predictor(df, 'Charge_Capacity (Ah)', desc), columns = ['Charge_Capacity (Ah)'])
    DC = pd.DataFrame(battery_predictor(df, 'Discharge_Capacity (Ah)', desc), columns = ['Discharge_Capacity (Ah)'])
    CE = pd.DataFrame(battery_predictor(df, 'Charge_Energy (Wh)', desc), columns = ['Charge_Energy (Wh)'])
    DE = pd.DataFrame(battery_predictor(df, 'Discharge_Energy (Wh)', desc), columns = ['Discharge_Energy (Wh)'])
    CEff = pd.DataFrame(battery_predictor(df, 'Coulombic_Efficiency (%)', desc), columns = ['Coulombic_Efficiency (%)'])
    EEff = pd.DataFrame(battery_predictor(df, 'Energy_Efficiency (%)', desc), columns = ['Energy_Efficiency (%)'])
    
    report = pd.concat([user_set,CC,DC,CE,DE,CEff,EEff],axis=1)
    return report


