from utils.Classifier_Results_Library import *
#from Config.Config import ControlSUB_ResultsDir
#
ControlSUB_ResultsDir = '/Users/User/PycharmProjects/MOX_Manduca_EAG_Analysis/Results'
ODEABEV_L = ['All']

for o in ODEABEV_L:
    # File_Dir=f'{ControlSUB_ResultsDir}/{o}/ClassifierResults/'
    # Save_Dir = f'{File_Dir}/Figures/'
    File_Dir=f'{ControlSUB_ResultsDir}/{o}/ClassifierResults/'
    Save_Dir = f'{File_Dir}/Figures/'

    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)

    name_map = {
        'linalool': 'Linalool',
        'floral': 'Floral Mix',
        'artcov': 'ArtCovid',
        'healthy': 'Healthy',
    }

    SVM_Results ='SVM_Results.pickle'
    RF_Results = 'RF_Results.pickle'

    SVM_DF = pickle_to_DF(f'{File_Dir}{SVM_Results}')
    RF_DF = pickle_to_DF(f'{File_Dir}{RF_Results}')
    print(SVM_DF['predicted_classes'][5])
    labels = [name_map[label] for label in SVM_DF['predicted_classes'][0] if label in name_map]

    print(len(labels))

    names=['SVM Results', 'RF Results']

    df=pd.concat([SVM_DF['accuracy_score'],RF_DF['accuracy_score'],
                  SVM_DF['accuracy_score'],RF_DF['accuracy_score']],axis=1,keys=names)



    SVM_CM = extract_CM(SVM_DF)

    RF_CM = extract_CM(RF_DF)
    #title='Lemon Oil, Limonene, 1-Octen-3-ol \n  Ylang Ylang, Benzylalcohol'
    #', '.join(labels)
    plot_CM(SVM_CM,labels,YROT=90, XROT=0, TITLE=None, SAVEDIR=f'{Save_Dir}SVM_')
    plot_CM(RF_CM,labels,YROT=90, XROT=0, TITLE=None, SAVEDIR=f'{Save_Dir}RF_')

    ViPlot(df,'Classifier Results',len(labels), INNER='box', DisplayMean=True, SAVEDIR=Save_Dir)
    #BoxPlot(df,'Classifier Results',len(labels))#,SAVEDIR=Save_Dir)