import os
import pandas as pd
import matplotlib.pyplot as plt

def EAG_df_build(DIR):

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

DIR = '/Data/Extracted_Waves/'

DF = EAG_df_build(DIR)






