import os.path
from Find_Solenoid import find_sol
from Format_CSV import Read_CSV_With_Col_Names
from config import TEST_CSV, RAW_DATA_OUTPATH
import matplotlib.pyplot as plt
import pandas as pd
#def Extract_mEAG(CSV, FREQUECY):
    # first we load the data into our variable abf
    # open a csv file containing a EAG wave

def Extract_Waves(CSV):
    BASENAME = os.path.basename(CSV)
    DF = Read_CSV_With_Col_Names(CSV)
    solenoid = DF['Solenoid']

    #the stimulus data is stored in "solenoid".
    #we need to identify when it is activated/inactivated
    sol = find_sol(solenoid)
    ni = len(sol)

    # store the channel to be processed in the variable temp
    TEMP = DF['Voltage']
    for i in range(0, ni):
        WAVES_DF = pd.DataFrame()

        #extract half second prior to solenoid activating and 4 seconds after solenoid activating
        #this decision was made due to variable lengths of solenoid activation
        WAVES_DF['Voltage'] = TEMP[sol[i][0] - 5: sol[i][0] + 40].values
        WAVES_DF['Solenoid'] = solenoid[sol[i][0] - 5: sol[i][0] + 40].values

        #create a save location
        OUTPATH_DIR = os.path.join(RAW_DATA_OUTPATH, 'Extracted_Waves/')
        os.makedirs(OUTPATH_DIR, exist_ok=True)

        #make a file name
        if '_1.csv' in BASENAME:
            OUTFILE_NAME = BASENAME.replace('_1.csv', '')
            OUTFILE_NAME = f'{OUTFILE_NAME}_wave{i}.csv'
            print(OUTFILE_NAME)
        #save every extracted wave from the original file as its own CSV
        WAVES_DF.to_csv(OUTPATH_DIR+OUTFILE_NAME)

Extract_Waves(TEST_CSV)