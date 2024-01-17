from utils.EAG_DataProcessing_Library import *
DATA_DIR='/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/Extracted_Waves/MOX/'

SAVE_DIR = '/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/DataFrames/MOX/'

df = MOX_DF_BUILD(DATA_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

df.to_csv(f'{SAVE_DIR}All.csv')