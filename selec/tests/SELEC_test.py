import unittest
import pandas as pd

#ohe_Ace unit test
class Test_oheace(unittest.TestCase):

    #We can test if the two row numbers before and after encoding are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        assert len(SELEC.ohe_ACE(df_battery)) == len(df_battery), 'The encoded row numbers are not identical'
        
#ohe_df unit test
class Test_ohedf(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        assert len(SELEC.ohe_dataframe(df_battery)) == len(df_battery), 'The encoded df has different rows'
        
#Data split unit test
class Test_split(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        train,test,X_train,y_train,X_test,y_test = SELEC.data_split(df_battery, 0.2, 'Discharge_Capacity (Ah)', 66)
        assert len(X_train) == len(df_battery) * 0.8, 'The dataset has not been splited correctly'
    
##kfolds unit test
class Test_kf(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        kf = SELEC.kfold(10,66)
        assert kf.random_state == 66, 'The kFolds has wrong random states'
        


##Data Scale Test
#Data scale unit test
class Test_scale(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        train,test,X_train,y_train,X_test,y_test = SELEC.data_split(df_battery, 0.2, 'Discharge_Capacity (Ah)', 66)
        X_train_scaled, X_test_scaled = SELEC.data_scale(X_train, X_test)
        assert len(X_test_scaled) == len(df_battery) * 0.2, 'The X_test data has not been properly scaled'
        
##GridSearchCv
#Hyperparameter unit test
class Test_hp(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        alg, n_neigh, weight = SELEC.grid_knn_hp(1, 51, df_battery, 'Discharge_Capacity (Ah)')
        alg2, n_neigh2, weight2 = SELEC.grid_knn_hp(1, 51, df_battery, 'Discharge_Capacity (Ah)')
        assert n_neigh == n_neigh2, 'Two sets of hyperparameters are not identical, try again'

##Model train
#Model unit test
class Test_ml(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        knn_train_RMSE_avg = SELEC.knn_train(df_battery, 'Discharge_Capacity (Ah)')
        knn_train_RMSE_avg2 = SELEC.knn_train(df_battery, 'Discharge_Capacity (Ah)')
        assert knn_train_RMSE_avg == knn_train_RMSE_avg, 'We have a different accuracy'
        
##Model test
#Model test unit test
class Test_mlt(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        knn_test_RMSE = SELEC.knn_test(df_battery, 'Discharge_Capacity (Ah)')
        knn_test_RMSE2 = SELEC.knn_test(df_battery, 'Discharge_Capacity (Ah)')
        assert knn_test_RMSE == knn_test_RMSE2, 'We have a different accuracy'

##Dataset Prepare
#X-set unit test
class Test_x(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        X_set = SELEC.X_set_in(df_battery)
        assert X_set.columns.all != ['anode', 'cathode', 'electrolyte', 'Cycle', 'temperature', 'discharge_crate'], 'X set has different inputs'
        
#X-set encode unit test
class Test_xen(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        X_set = SELEC.X_set_in(df_battery)
        X_set_EN = SELEC.X_set_en(df_battery)
        assert len(X_set_EN) == len(X_set), 'The encoded X set has different rows'
        
##Predictor unit test
#predictor prep unit test
class Test_pp(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        X_bat, y_bat = SELEC.df_prep(df_battery, 'Discharge_Capacity (Ah)')
        assert len(X_bat) == len(y_bat), 'The x has different row values to y'
        
#predictor unit test
class Test_ppy(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        y_pred = SELEC.battery_predictor(df_battery, 'Discharge_Capacity (Ah)')
        y_pred2 = SELEC.battery_predictor(df_battery, 'Discharge_Capacity (Ah)')
        assert y_pred.all() == y_pred2.all(), 'The two predictions are diffrent'
        
##Report unit test
#Report unit test
class Test_report(unittest.TestCase):
    #We can test if the two rows we select are the same
    def test_failure(self):
        import SELEC
        df_battery = pd.read_csv('Battery_Dataset.csv')
        report = SELEC.report_gen(df_battery)
        X_set = SELEC.X_set_in(df_battery)
        assert len(report) == len(X_set), 'The ouput has diffrent dimensions'