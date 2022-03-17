import unittest
import pandas as pd

#df_battery = pd.read_csv('../data/Battery_Dataset.csv')

#ohe_Ace unit test
class Test_oheace(unittest.TestCase):

    #We can test if the two row numbers before and after encoding are the same
    def test_failure(self):
        from selec.dataprepare.dataprep import oheACE 
        assert len(dataprep.ohe_ACE(df_battery)) == len(df_battery), 'The encoded row numbers are not identical'
        
#ohe_df unit test
class Test_ohedf(unittest.TestCase):

    #We can test if the two rows we select are the same
    def test_failure(self):
        from selec.dataprepare.dataprep import ohe_dataframe
        assert len(dataprep.ohe_dataframe(df_battery)) == len(df_battery), 'The encoded df has different rows'

#ohe_encoded_values unit test
class Test_encode(unittest.TestCase):
    #we can test if the all the A1 values equals to one and E3 values equals to zero
    def test_A1values():
        from selec.dataprepare.dataprep import ohe_dataframe
        df_enbattery = ohe_dataframe(df_battery)
        A1 = df_enbattery.iloc[:,0:1]
        for i in range(len(A1)):
            assert Ai[i] == 1 , 'wrong values for A1'
    def test_E3values():
        from selec.dataprepare.dataprep import ohe_dataframe
        df_enbattery = ohe_dataframe(df_battery)
        E3 = df_enbattery.iloc[:,6:7]
        for j in range(len(E3)):
            assert E3[1] == 0, 'wrong values for E3'
        
