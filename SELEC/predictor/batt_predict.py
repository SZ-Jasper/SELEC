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

from batt_descriptor.batt_describe import *
from model.knn import *


def X_set_in(df_battery):
    """ DOC STRING TO BE ADDED """
    ac = df_battery['anode'].unique()
    cc = df_battery['cathode'].unique()
    ec = df_battery['electrolyte'].unique()
    cycle = df_battery['Cycle'].unique()
    tc = df_battery['temperature'].unique()
    cr = df_battery['discharge_crate'].unique()

    X_temp = []
    for a in range(len(ac)):
        for b in range(len(cc)):
            for c in range(len(ec)):
                for d in range(len(cycle)):
                    for e in range(len(tc)):
                        for f in range(len(cr)):
                            X_temp.append([ac[a],cc[b],ec[c],
                                           cycle[d],tc[e],cr[f]])
                        
    column_names = ['anode','cathode','electrolyte',
                    'Cycle','temperature','discharge_crate']
    X_set = pd.DataFrame(X_temp, columns = column_names)
        
    return X_set


def X_set_en(df):
    """ DOC STRING TO BE ADDED """
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


def df_prep(df, output):
    """ DOC STRING TO BE ADDED """
    df_encoded_ACE = ohe_ACE(df)
    df_enbattery = ohe_dataframe(df)
    
    X_bat = df_enbattery.loc[:,['A1','C1','C2','C3','E1','E2','E3',
                                'Cycle','temperature','discharge_crate']]
    y_bat = df_enbattery.loc[:,[output]]
    
    return X_bat, y_bat


def battery_predictor(df, output):
    """ DOC STRING TO BE ADDED """
    X_bat, y_bat = df_prep(df, output)
    X_set = X_set_en(df)
    X_bat_scaled, X_set_scaled = data_scale(X_bat, X_set)
    
    best_knn_hp = grid_knn_hp(1, 51, df, output)
    alg = best_knn_hp[0]
    n_neigh = best_knn_hp[1]
    weight = best_knn_hp[2]

    np.random.seed(66)
    KNN_model =KNeighborsRegressor(algorithm=alg, 
                                   n_neighbors=n_neigh, 
                                   weights=weight)
    
    KNN_model.fit(X_bat_scaled,y_bat)
    y_predict=KNN_model.predict(X_set_scaled)
    return y_predict


def report_gen(df):
    """ DOC STRING TO BE ADDED """
    in_set = X_set_in(df)
    
    CC = pd.DataFrame(battery_predictor(df, 'Charge_Capacity (Ah)'), 
                      columns = ['Charge_Capacity (Ah)'])
    DC = pd.DataFrame(battery_predictor(df, 'Discharge_Capacity (Ah)'), 
                      columns = ['Discharge_Capacity (Ah)'])
    CE = pd.DataFrame(battery_predictor(df, 'Charge_Energy (Wh)'), 
                      columns = ['Charge_Energy (Wh)'])
    DE = pd.DataFrame(battery_predictor(df, 'Discharge_Energy (Wh)'), 
                      columns = ['Discharge_Energy (Wh)'])
    CEff = pd.DataFrame(battery_predictor(df, 'Coulombic_Efficiency (%)'), 
                        columns = ['Coulombic_Efficiency (%)'])
    EEff = pd.DataFrame(battery_predictor(df, 'Energy_Efficiency (%)'),
                        columns = ['Energy_Efficiency (%)'])
    
    report = pd.concat([in_set,CC,DC,CE,DE,CEff,EEff],axis=1)
    return report