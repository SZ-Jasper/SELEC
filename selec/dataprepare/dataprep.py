import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def ohe_ACE(df):
    """ One Hot Encoding dataset """
    ohe = OneHotEncoder()
    ACE = df.loc[:,['anode','cathode','electrolyte']]
    ACE = ohe.fit_transform(ACE)
    df_encoded_ACE = pd.DataFrame(ACE.toarray())
    return df_encoded_ACE

def ohe_dataframe(df):  
    """ Add one hot encoded data to dataset """
    df_ACE_col_name = ['A1','C1','C2','C3','E1','E2','E3']
    df_encoded_ACE = ohe_ACE(df)
    for i in range(len(df_encoded_ACE.columns)):
        df_encoded_ACE = df_encoded_ACE.rename({df_encoded_ACE.columns[i]: 
                                                df_ACE_col_name[i]}, axis=1) 
    df_ohe = pd.concat([df_encoded_ACE, df],axis=1)
    return df_ohe


### Split Data
def data_split (df, test_ratio, output, seed):
    """ split data """
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

### Scale train and test sets
def data_scale(X_train, X_test):
    """ Standardize data with scalar """
    scaler = StandardScaler(with_mean=True,with_std=True)
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    scaler.fit(X_test)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    return X_train_scaled,X_test_scaled