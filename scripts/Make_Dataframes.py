from utils.EAG_DataProcessing_Library import *
DATA_DIR='/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/' \
         'MOX_Manduca_EAG_Analysis/Data/Extracted_Waves/'

SAVE_DIR = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/' \
           'MOX_Manduca_EAG_Analysis/Data/DataFrames/'

df = EAG_DF_BUILD(DATA_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

df.to_csv(f'{SAVE_DIR}All.csv')
