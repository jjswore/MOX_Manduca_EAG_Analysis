import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os


def MOX_PCA(DATA, SAVEDIR, CONC, DUR, ODORS, OA):
    concentration = CONC
    odors = ODORS
    OdorAbreve = OA
    data = DATA
    SaveDir = f'{SAVEDIR}/PCA/'

    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    global DF
    global All_DF
    if isinstance(data, str):
        # Additional check to see if it ends with '.csv' if needed
        if data.endswith('.csv'):
            DF = pd.read_csv(data, index_col=0,dtype={'concentration': 'string', 'duration':'string'})
        else:
            print("String provided is not a path to a CSV file.")
            return

        # If data is already a dataframe
    elif isinstance(data, pd.DataFrame):
        DF = DATA

    else:
        print("Unsupported data format.")
        return
    global DF2
    # load our data frame into the program
    # ?!\d captures exact matches so you only match 1 not 10.
    DF2 = DF[DF['concentration'].str.contains(f'(?<!\d){concentration}(?!\d)')]
    All_DF = DF2[DF2['duration'].str.contains(f'(?<!\d){DUR}(?!\d)')]
    All_DF = All_DF[All_DF['label'].str.contains(f'(?<!\d){odors}(?!\d)')]

    # convert the data frame into a usable format for scaling
    # leave out the metadata columns
    All_DF2 = All_DF.iloc[:, :-6].convert_dtypes(float).astype(float)

    # scale the data to calculate the principal componenets
    All_DF_Scaled = pd.DataFrame(StandardScaler().fit_transform(All_DF2),
                                 columns=All_DF2.columns,
                                 index=All_DF2.index)

    # set the PCA parameters
    PCA_set = PCA(n_components=10)

    # Find the principal components of your dataset
    All_DF_PCAResults = PCA_set.fit_transform(All_DF_Scaled)
    print(PCA_set.explained_variance_ratio_)

    # Save our PCA object
    reader = open(f'{SaveDir}{OdorAbreve}_PCA.pickle', 'wb')
    pickle.dump(obj=PCA_set, file=reader)
    reader.close()

    # Store the principal components in a dataframe
    All_DF_PCA_DF = pd.DataFrame(data=All_DF_PCAResults, index=All_DF_Scaled.index)
    for x, y in zip(range(10), range(1, 11)):
        All_DF_PCA_DF.rename({x: f'PC {y}'}, axis=1, inplace=True)
    All_DF_PCA_DF = pd.concat([All_DF_PCA_DF,All_DF.iloc[:,-6:]], axis=1 )

    # save your PC datafram
    All_DF_PCA_DF.to_csv(f'{SaveDir}/{OdorAbreve}_PCA.csv')

    return All_DF_PCA_DF, PCA_set

DATA_FILE='/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/DataFrames/All.csv'
SAVE_DIR='/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Data/Results/MOX/'

PCA_DF, PCA_OBJ = MOX_PCA(DATA=DATA_FILE, SAVEDIR=SAVE_DIR, CONC='1k|100k', ODORS='floral|linalool|healthy|artcov', DUR='02|1|5|10', OA='All')