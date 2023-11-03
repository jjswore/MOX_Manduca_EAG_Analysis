from utils.EAG_SIngleChannel_DataProcessing_Library import *

df = EAG_df_build('/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/'
                  'Data/Normalized/NoFilt/Extracted_Waves')
save = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/' \
       'Data/Normalized/NoFilt/Dataframes/'
if not os.path.exists(save):
    os.makedirs(save)
df.to_csv(f'{save}All_Odors.csv')
