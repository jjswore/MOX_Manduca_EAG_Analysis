import os
import pandas as pd
import matplotlib.pyplot as plt

DIR = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/MOX_Manduca_EAG_Analysis/Data/Raw/0727231505M1A1_1k_artCov_00_1.csv'
#define column names

def Read_CSV_With_Col_Names(FilePath):
    column_names = ['Time', 'Voltage', 'Solenoid']
    df = pd.read_csv(FilePath, header=None, names=column_names)
    return df


