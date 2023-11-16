import os

from utils.EAG_DataProcessing_Library import *

DATA_DIR = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/' \
           'MOX_Manduca_EAG_Analysis/Data/Raw/'
CSV_SAVEDIR='/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/' \
            'MOX_Manduca_EAG_Analysis/Data/'
for file in os.listdir(DATA_DIR):
    if '.csv' in file:
        Extract_Waves(DATA_DIR+file, CSV_SAVEDIR)


