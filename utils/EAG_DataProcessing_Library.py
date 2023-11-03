import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import glob
from scipy.signal import butter, lfilter
from scipy import optimize

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

def get_EAGs(data):
    """
        Extracts the timeseries data for each channel in a given dataset.

        Args:
            data (list): A 2D list representing the dataset containing EAG data.
                         The data should be organized such that each row represents
                         a time point, and each column represents a different channel.
                         The first row contains the channel names, and the following
                         rows contain the corresponding data values.

        Returns:
            dict: A dictionary where each key is a channel name, and the corresponding
                  value is a list of data points for that channel.

        Raises:
            None

        Example Usage:
            data = [
                ["t", "EAG1", "EAG4", "EAG3", "EAG2","Solenoid"],
                ["0", "0.1", "0.2", "0.3"],
                ["1", "0.4", "0.5", "0.6"],
                ["2", "0.7", "0.8", "0.9"]
            ]
            eag_dict = get_EAGs(data)
            print(eag_dict)
            # Output: {'Channel1': [0.1, 0.4, 0.7], 'Channel2': [0.2, 0.5, 0.8], 'Channel3': [0.3, 0.6, 0.9]}

        """
    chList = data[0][:]

    # Find the indices of the channels in the first row
    chIndices = [i for i, val in enumerate(data[0]) if val in chList]

    # Create an empty dictionary to store the channels and their data
    channels_data = {}

    # Loop through each channel index
    for chIndex in chIndices:
        # Get the channel name using the index
        ch = chList[chIndex]

        # Extract the data for the current channel
        channel_data = [float(data[x][chIndex]) for x in range(1, len(data))]
        # Calculate the baseline by taking the average of rows 50 to 250
        baseline = sum(channel_data[49:250]) / (250 - 50 + 1)

        # Subtract the baseline from the entire column
        baseline_subtracted_data = [value - baseline for value in channel_data]

        # Store the channel and its data in the dictionary
        channels_data[ch] = baseline_subtracted_data
    return channels_data

def Extract_mEAG(FILE, record_channels=['EAG2']):
    # first we load the data into our variable abf
    # open a csv file containing a EAG wave
    #with open(FILE, newline='') as f:
    with open(FILE, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    EAGs_dict = get_EAGs(data)
    solenoid = EAGs_dict['Solenoid']
    #the stimulus data is stored in "solenoid". we need to identify when it is activated/inactivated
    sol = find_sol(solenoid)
    ni = len(sol)

    # extract a 4 second window centered on the solenoid for each wave (this means three waves per channel
    # Each key in the dictionary is either the time 't', or a channel 'EAG1', 'EAG4','EAG3','EAG2','Solenoid'
    # the values for each channel with be three lists 5 seconds in length (400 points)

    for key, value in EAGs_dict.items():
        if key not in record_channels:
            continue
        #print(key)
        intervals = []
        # store the channel to be processed in the variable temp
        TEMP = EAGs_dict[key]
        for i in range(0, ni):
            interval = TEMP[sol[i][0] - 50: sol[i][1] + 400]

            intervals.append(interval)
        EAGs_dict[key] = intervals
        #print(EAGs_dict[key])

    filtered_dict = {key: value for key, value in EAGs_dict.items() if key in record_channels}
    return filtered_dict

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

def process_data(DL=[], record_channels=['EAG1','EAG2','EAG3','EAG4'], savedir=None, RETURN='Save'):
    """
        Process data from DList and save the processed data to savedir.

        Parameters:
        DList (list): A list of directories containing the data to process.
        norm (str): Normalization method to use. Possible values: 'YY', 'Yes', False. Default is 'YY'. This normalizes
                    data to the strongest odorant ylangylang.
        sub (bool): Whether to subtract channel 1 from channel 2. Default is False.
        savedir (str): Directory to save processed data to. Default is ''.

        Returns: None
        """
    print('Starting Data_Processing')

    SAVEDIR = savedir
    DList=[(subdir +'/') for directory in DL for subdir in get_subdirectories(directory)]

    print(f'this is the DLIST: {DList}')
    for D in DList:
        print('beginning ', D)
        f1 = [f.path for f in os.scandir(D)
              if 'DS_Store' not in os.path.basename(f)]
        # seperate the data into experimental and control lists
        ctrl = [x for x in f1 if 'mineraloil' in os.path.basename(x.lower()) or 'compressedair' in os.path.basename(x.lower())]
        print(f'this is the control{ctrl}')
        exp = [x for x in f1 if 'mineraloil' not in os.path.basename(x.lower()) or 'compressedair' not in os.path.basename(x.lower())]
        YY = [x for x in f1 if 'ylangylang' in os.path.basename(x.lower())]


        # Extract the each individual wave and subtract the miniral oil control
        for data in exp:
            # print(data,control)
            n = os.path.basename(data)
            print(n, 'is an experiment')
            VOC = n.split("_")[4]
            if n.split("_")[2] == 's':
                delivery = 'Syringe'
            elif n.split("_")[2] == 'p':
                delivery = 'Pipette'
            #create the directory where waves will be saved
            DIR = f'{SAVEDIR}{delivery}/{VOC}/'
            n = namer(data)
            Odor = Extract_mEAG(data,record_channels, Butter[0], Butter[1], BF=B_filt)

            for x in range(0, 3):
                # subtract the control

                for key in Odor.keys():
                    Odor[key][x] = [a - b for a, b in zip(Odor[key][x], [np.mean(values) for values in zip(*[wave[:500] for wave in CTRL[key]])])]

            if RETURN == 'SAVE':
                SaveData(Odor, directory=DIR, name=n)


            elif RETURN == 'PLOT':
                for key in Odor.keys():
                    for w in range(3):
                        i = input('do you want to plot')
                        if i.lower() == 'yes':
                            plt.plot(Odor[key][w])
                            plt.title(f'{VOC}{key} wave{w}')
                            plt.ylim(-1.5,1.5)
                            plt.show()
                        else:
                            break
            print('finished')
