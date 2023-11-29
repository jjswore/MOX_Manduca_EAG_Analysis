import os

from utils.EAG_DataProcessing_Library import *

DATA_DIR = '/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/MOX_Raw/'
CSV_SAVEDIR = '/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/'
for file in os.listdir(DATA_DIR):
    if '.csv' in file:
        Extract_MOX_Waves(DATA_DIR+file, CSV_SAVEDIR)


