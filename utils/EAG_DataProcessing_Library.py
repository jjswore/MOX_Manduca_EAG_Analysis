import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import glob
from scipy.signal import butter, lfilter
from scipy import optimize

#read in CSV and add column headers
def Read_CSV_With_Col_Names(FilePath):
    column_names = ['Time', 'Voltage', 'Solenoid']
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


def Extract_Waves(CSV, SAVE=True):
    BASENAME = os.path.basename(CSV)
    DF = Read_CSV_With_Col_Names(CSV)
    solenoid = DF['Solenoid']

    # the stimulus data is stored in "solenoid".
    # we need to identify when it is activated/inactivated
    sol = find_sol(solenoid)
    ni = len(sol)

    # store the channel to be processed in the variable temp
    TEMP = DF['Voltage']
    for i in range(0, ni):
        WAVES_DF = pd.DataFrame()

        # extract half second prior to solenoid and 4 seconds after
        # store in the the data frame
        WAVES_DF['Voltage'] = TEMP[sol[i][0] - 5: sol[i][1] + 40].values
        WAVES_DF['Solenoid'] = solenoid[sol[i][0] - 5: sol[i][1] + 40].values

        # create a save location
        OUTPATH_DIR = os.path.join(RAW_DATA_OUTPATH, 'Extracted_Waves/')
        os.makedirs(OUTPATH_DIR, exist_ok=True)

        # make a file name
        if '_1.csv' in BASENAME:
            OUTFILE_NAME = BASENAME.replace('_1.csv', '')
            OUTFILE_NAME = f'{OUTFILE_NAME}_wave{i}.csv'
            print(OUTFILE_NAME)
        # save every extracted wave from the original file as its own CSV
        WAVES_DF.to_csv(f'{OUTPATH_DIR}{BASENAME}wave_{i}.csv')


def name_con(f):
    # This will splits the basename of the file on "_" to find the concentration in the file name
    #'dateMoth#SexAntenna#_line_deliverymethod_odor_trial' 062623M1fA1_OR6KO_p_linalool_1
    tn = os.path.basename(f).split("_", 3)
    name = f'{tn[0]}{tn[3]}'
    return name

def SaveData(data, directory, name):  # data is an numpy array
    # saves nested array of multichannel data to a

    Dir = directory
    if not os.path.isdir(Dir):
        os.makedirs(Dir)

    for key in data:
        for wave_idx, wave_data in enumerate(data[key]):
            with open(f'{Dir}/{name}_{key}_wave{wave_idx}.csv', 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(wave_data)


def namer(f):
    #removes the "." from the file name
    n = os.path.basename(f)
    tn = n.split(".")

    return tn[0]

def open_wave(FILE):
    #open a csv file containing a EAG wave
    with open(FILE, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    f.close()
    l = data[0]
    l = list(map(float, l))
    return l

def csv_plot(FILE, NAME, SDir, SAVE=True):
    #plot a csv file
    t = open_wave(FILE)
    # n=NAME.split("\\")
    plt.title(label=NAME, size=10)
    plt.plot(t)
    plt.ylim(-1,1)
    if SAVE == True:
        plt.savefig(SDir+NAME+".svg")
        plt.savefig(SDir+NAME+".jpg")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def get_subdirectories(directory):
    subdirectories = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            subdirectories.append(path)
    return subdirectories


