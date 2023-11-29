import numpy as np
import pandas as pd
import os


#read in CSV and add column headers
def Read_CSV_With_Col_Names(FilePath):
    column_names = ['Time', 'Voltage', 'Solenoid']
    df = pd.read_csv(FilePath, header=None, names=column_names)
    return df

#read in MOX CSV and add column headers
def Read_MOX_CSV_With_Col_Names(FilePath):
    column_names = ['Resistance', 'Solenoid']
    df = pd.read_csv(FilePath, header=None, names=column_names)
    return df

def find_sol(data):
    """
    This function takes an array of data, calculates the difference between consecutive 
    elements, and then finds the first indices of series where the difference is either 
    less than -0.6 or greater than 0.6.

    Arguments:
    data -- a list or array-like object of numerical values

    Returns:
    sol -- a list of tuples, where the first element of each tuple is an index where 
           the difference is greater than 0.6, and the second element is an index where 
           the difference is less than -0.6.
    """
    # Calculate the difference between consecutive elements in the array
    SolFall = np.diff(data)

    # Initialize the lists to store the indices and the flags
    NI, PI = [], []
    flag_ni, flag_pi = False, False

    # Iterate through the SolFall array starting from the 6th element
    for i, v in enumerate(SolFall[5:]):
        # If the difference is less than -0.6 and the last data point did not meet this condition
        if v < -0.6 and not flag_ni:
            # Add the current index to the NI list
            NI.append(i)
            # Set the flag to True to indicate that the current data point meets the condition
            flag_ni = True
        elif v >= -0.6:
            # If the difference is not less than -0.6, reset the flag to False
            flag_ni = False

        # If the difference is more than 0.6 and the last data point did not meet this condition
        if v > 0.6 and not flag_pi:
            # Add the current index to the PI list
            PI.append(i)
            # Set the flag to True to indicate that the current data point meets the condition
            flag_pi = True
        elif v <= 0.6:
            # If the difference is not more than 0.6, reset the flag to False
            flag_pi = False

    # Create a list of tuples where each tuple contains an element from PI and an element from NI
    test = zip(PI, NI)
    # Convert the zip object to a list
    sol = list(test)

    # Return the list of tuples
    return sol
#fjkdikk
def Extract_Waves(CSV, SAVEDIR):
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
        OUTPATH_DIR = os.path.join(SAVEDIR, 'Extracted_Waves/')
        os.makedirs(OUTPATH_DIR, exist_ok=True)

        #make a file name
        OUTFILE_NAME = BASENAME.replace('.csv','')
        OUTFILE_NAME = f'{OUTFILE_NAME}_wave{i}.csv'
        print(OUTFILE_NAME)

        #save every extracted wave from the original file as its own CSV
        WAVES_DF.to_csv(OUTPATH_DIR+OUTFILE_NAME)

#for MOX data extraction
def Extract_MOX_Waves(CSV, SAVEDIR):
    BASENAME = os.path.basename(CSV)
    DF = Read_MOX_CSV_With_Col_Names(CSV)
    solenoid = DF['Solenoid']

    ni = int(len(solenoid) / 10)

    # store the channel to be processed in the variable temp
    TEMP = DF['Resistance']
    for i in range(0, ni):
        WAVES_DF = pd.DataFrame()

        #extract half second prior to solenoid activating and 4 seconds after solenoid activating
        #this decision was made due to variable lengths of solenoid activation
        WAVES_DF['Resistance'] = TEMP[i*10 : i*10+10].values
        WAVES_DF['Solenoid'] = solenoid[i*10 : i*10+10].values

        #create a save location
        OUTPATH_DIR = os.path.join(SAVEDIR, 'Extracted_Waves/MOX/')
        os.makedirs(OUTPATH_DIR, exist_ok=True)

        #make a file name
        OUTFILE_NAME = BASENAME.replace('.csv','')
        OUTFILE_NAME = f'{OUTFILE_NAME}_wave{i}.csv'
        print(OUTFILE_NAME)

        #save every extracted wave from the original file as its own CSV
        WAVES_DF.to_csv(OUTPATH_DIR+OUTFILE_NAME)

def EAG_DF_BUILD(DIR):

    F_List = os.listdir(DIR)

    #create lists to store meta data for each file
    master = []
    NAME_L = []
    LABEL_L = []
    CONCENTRATION_L = []
    DATE_L = []
    TRIAL_L = []
    WAVE_L = []



    for file in F_List:
        print(file)
        # seperate the metadata into individual strings
        n = os.path.basename(file.lower()).split("_")
        n[4] = n[4].replace('.csv', '')
        name = n[0] + n[1] + n[2] + n[3] + n[4]

        #extract the metadata
        date = n[0]
        concentration = n[1]
        label = n[2]
        trial = n[3]
        wave_number = n[4]

        #read in the data
        x = pd.read_csv(DIR+file, index_col=0)['Voltage']

        #store the metadata
        master.append(x)
        NAME_L.append(name)
        LABEL_L.append(label)
        CONCENTRATION_L.append(concentration)
        DATE_L.append(date)
        TRIAL_L.append(trial)
        WAVE_L.append(wave_number)

    #place the data into a data frame and create collumns for meta data at end of row
    master_df = pd.DataFrame(dict(zip(NAME_L, master)), index=[x for x in range(0, len(x))])
    master_df = master_df.T
    master_df['label'] = LABEL_L
    master_df['concentration'] = CONCENTRATION_L
    master_df['date'] = DATE_L
    master_df['trial'] = TRIAL_L
    master_df['wave_number'] = WAVE_L

    return master_df

def MOX_DF_BUILD(DIR):

    F_List = os.listdir(DIR)

    #create lists to store meta data for each file
    master = []
    NAME_L = []
    LABEL_L = []
    CONCENTRATION_L = []
    DATE_L = []
    DURATION_L = []
    TRIAL_L = []
    WAVE_L = []



    for file in F_List:
        print(file)
        # seperate the metadata into individual strings
        n = os.path.basename(file.lower()).split("_")
        n[5] = n[5].replace('.csv', '')
        name = n[0] + n[1] + n[2] + n[3] + n[4] + n[5]

        #extract the metadata
        date = n[0]
        concentration = n[1]
        label = n[2]
        duration = n[3]
        trial = n[4]
        wave_number = n[5]

        #read in the data
        x = pd.read_csv(DIR+file, index_col=0)['Resistance']

        #store the metadata
        master.append(x)
        NAME_L.append(name)
        LABEL_L.append(label)
        CONCENTRATION_L.append(concentration)
        DATE_L.append(date)
        DURATION_L.append(duration)
        TRIAL_L.append(trial)
        WAVE_L.append(wave_number)

    #place the data into a data frame and create collumns for meta data at end of row
    master_df = pd.DataFrame(dict(zip(NAME_L, master)), index=[x for x in range(0, len(x))])
    master_df = master_df.T
    master_df['label'] = LABEL_L
    master_df['concentration'] = CONCENTRATION_L
    master_df['date'] = DATE_L
    master_df['trial'] = TRIAL_L
    master_df['duration'] = DURATION_L
    master_df['wave_number'] = WAVE_L

    return master_df











