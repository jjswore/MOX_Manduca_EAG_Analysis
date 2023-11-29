import os
import pandas as pd
import matplotlib.pyplot as plt

def MOX_df_build(DIR):

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