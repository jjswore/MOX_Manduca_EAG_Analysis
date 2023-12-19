from utils.EAG_Classifier_Library import *  # Importing all functions from EAG_Classifier_Library
import pandas as pd
import os

CLASSES = [] #these are the odors you are trying to classify 'limonene|lemonoil'
CONCS = [] #these are the concentrations used '1k' or '1k|10k'
DURS = [] #stimulus duration used '1|5' would use both 1 and 5 sec durations

# # Read in the dataset as a Dataframe
DF = pd.read_csv(f'PATH/TO/YOUR/DATA.csv',
                     index_col=0, dtype={'concentration': 'string'})

# Extract a specific range of columns from the testing DataFrame
#.iloc[row 0 : Row n, Column 250 : Column 5500] adjust to fit your needs
Data_For_Model_Testing = DF.iloc[:, 250:5500]
# Extract the last three columns (meta-data) from the filtered PCA DataFrame
#.iloc[row 0 : Row n, - FirstColumnofMetaData:]
# the '-' indicates you'll take all data from there to the end
Meta_Data = DF.iloc[:, -3:]

# Concatenate the time series data and meta-data
Test_DF = pd.concat([Data_For_Model_Testing, Meta_Data], axis=1)

# Define the directory to save classifier results
Save_Directory = f'/PATH/TO/SAVE/DIREDTORY/'

# Create the directory if it doesn't exist
if not os.path.exists(Save_Directory):
    os.makedirs(Save_Directory)

# Extract unique odor labels and concentrations from Test_DF
Test_DF = Test_DF[Test_DF['label'].str.contains(CLASSES)]
Test_DF = Test_DF[Test_DF['concentration'].str.contains(CONCS)]
Test_DF = Test_DF[Test_DF['durations'].str.contains(DURS)]

# Perform SVM classification
print(f'beginning SVM...')
SVM_Results = SVM_model_Testing(data=[Test_DF], recall_class='1octen3ol', repeats=100)
# Save SVM results using pickle
pickle_Saver(savedir=Save_Directory, ext='SVM_Results', data=SVM_Results)

# Perform Random Forest classification
print(f'beginning Random Forest')
RF_results = RF_model_Testing(data=[Test_DF], recall_class='1octen3ol', repeats=100)
# Save Random Forest results using pickle
pickle_Saver(savedir=Save_Directory, ext='RF_Results', data=RF_results)
