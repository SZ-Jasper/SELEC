import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import OneHotEncoder


# you're gonna have to change the location
df_battery = pd.read_csv('../data/Battery_Dataset.csv') 


def ohe_ACE(df):
    """ DOC STRING TO BE ADDED """
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ACE = ohe.fit_transform(ACE)
    df_encoded_ACE = pd.DataFrame(ACE.toarray())
    return df_encoded_ACE


def ohe_dataframe(df):
    """ DOC STRING TO BE ADDED """
    df_ACE_col_name = ['A1','C1','C2','C3','E1','E2','E3']
    df_encoded_ACE = ohe_ACE(df)
    for i in range(len(df_encoded_ACE.columns)):
        df_encoded_ACE = df_encoded_ACE.rename({df_encoded_ACE.columns[i]: 
                                                df_ACE_col_name[i]}, axis=1) 
    df_ohe = pd.concat([df_encoded_ACE, df],axis=1)
    return df_ohe