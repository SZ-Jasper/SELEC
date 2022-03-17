import unittest
import pandas as pd
from selec.dataprepare.dataprep import oheACE

df_battery = pd.read_csv('../data/Battery_Dataset.csv')

#ohe_Ace unit test
class Test_oheace(unittest.TestCase):

    #We can test if the two row numbers before and after encoding are the same
    def test_failure(self):
        assert len(ohe_ACE(df_battery)) == len(df_battery), 'The encoded row numbers are not identical'
        
#ohe_df unit test
class Test_ohedf(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        assert len(ohe_dataframe(df_battery)) == len(df_battery), 'The encoded df has different rows'

#ohe_encoded_values unit test
df_encoded_ACE = ohe_ACE(df_battery)
A1 = df_enbattery.iloc[:,0:1]
E3 = df_enbattery.iloc[:,6:7]
class Test_encode(unittest.TestCase):
    #we can test if the all the A1 values equals to one and E3 values equals to zero
    def test_A1values():
        for i in range(len(A1)):
            assert Ai[i] == 1 , 'wrong values for A1'
    def test_E3values():
        for j in range(len(E3)):
            assert E3[1] == 0, 'wrong values for E3'
        
##Data Split
#Data split unit test
class Test_split(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, 'Discharge_Capacity (Ah)', 66)
        assert len(X_train) == len(df_battery) * 0.8, 'The dataset has not been splited correctly'
    
##kfolds unit test
class Test_kf(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        kf = kfold(10,66)
        assert kf.random_state == 66, 'The kFolds has wrong random states'

##Data Scale Test
#Data scale unit test
class Test_scale(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        train,test,X_train,y_train,X_test,y_test = data_split(df_battery, 0.2, 'Discharge_Capacity (Ah)', 66)
        X_train_scaled, X_test_scaled = data_scale(X_train, X_test)
        X_test_scaled
        assert len(X_test_scaled) == len(df_battery) * 0.2, 'The X_test data has not been properly scaled'
        
##GridSearchCv
#Hyperparameter unit test
class Test_hp(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        alg, n_neigh, weight = grid_knn_hp(1, 51, df_battery, 'Discharge_Capacity (Ah)')
        alg2, n_neigh2, weight2 = grid_knn_hp(1, 51, df_battery, 'Discharge_Capacity (Ah)')
        assert n_neigh == n_neigh2, 'Two sets of hyperparameters are not identical, try again'

##Model train
#Model unit test
class Test_ml(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        knn_train_RMSE_avg = knn_train(df_battery, 'Discharge_Capacity (Ah)')
        knn_train_RMSE_avg2 = knn_train(df_battery, 'Discharge_Capacity (Ah)')
        assert knn_train_RMSE_avg == knn_train_RMSE_avg, 'We have a different accuracy'
        
##Model test
#Model test unit test
class Test_mlt(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        knn_test_RMSE = knn_test(df_battery, 'Discharge_Capacity (Ah)')
        knn_test_RMSE2 = knn_test(df_battery, 'Discharge_Capacity (Ah)')
        assert knn_test_RMSE == knn_test_RMSE2, 'We have a different accuracy'

##Dataset Prepare
#X-set unit test
class Test_x(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        X_set = X_set_in(df_battery)
        assert X_set.columns.all != ['anode', 'cathode', 'electrolyte', 'Cycle', 'temperature', 'discharge_crate'], 'X set has different inputs'
        
#X-set encode unit test
class Test_xen(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        X_set_EN = X_set_en(df_battery)
        assert len(X_set_EN) == len(X_set), 'The encoded X set has different rows'
        
##Predictor unit test
#predictor prep unit test
class Test_pp(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        X_bat, y_bat = df_prep(df_battery, 'Discharge_Capacity (Ah)')
        assert len(X_bat) == len(y_bat), 'The x has different row values to y'
        
#predictor unit test
class Test_ppy(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        y_pred = battery_predictor(df_battery, 'Discharge_Capacity (Ah)')
        y_pred2 = battery_predictor(df_battery, 'Discharge_Capacity (Ah)')
        assert y_pred.all() == y_pred2.all(), 'The two predictions are diffrent'
        
##Report unit test
#Report unit test
class Test_report(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        report = report_gen(df_battery)
        assert len(report) == len(X_set), 'The ouput has diffrent dimensions'