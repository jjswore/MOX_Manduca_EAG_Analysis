from utils.EAG_DataProcessing_Library import *

df = EAG_DF_BUILD()
save =

if not os.path.exists(save):
    os.makedirs(save)

df.to_csv(f'{save}All_Odors.csv')
