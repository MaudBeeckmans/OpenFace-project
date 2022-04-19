# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:54:45 2022

@author: maudb
"""

import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def import_data(pp_numbers = np.array([["1", "5"],["6", "10"]]), datafile_path = None): 
    count = 0
    for files in pp_numbers: 
        data_path = os.path.join(datafile_path, str('data_processed_concat' + files[0] + 'to' + 
                                                  files[-1] + '_AUstatic.rds'))
        data = pyreadr.read_r(data_path) # also works for RData
        data_df = data[None]
        if count == 0: all_data_df = data_df
        else: all_data_df = pd.concat([all_data_df, data_df])
        count += 1
        all_data_df.index = np.arange(0, all_data_df.shape[0], 1)
    return all_data_df

#%% Niet nodig, behavioural data is al toegevoegd aan de OpenFace output data!
def import_behaviouraldata(pp_numbers = np.arange(1, 11, 1), datafile_path = None): 
    pp_count = 0
    for pp in pp_numbers: 
        behavioural_file = os.path.join(datafile_path, "Participant{}".format(pp), "output_participant{}.csv".format(pp))
        behavioural_data_pp = pd.read_csv(behavioural_file)
        behavioural_data_pp['pp_number'] = pp
        if pp_count == 0: behavioural_data = behavioural_data_pp
        else: behavioural_data = pd.concat([behavioural_data, behavioural_data_pp])
        pp_count += 1
    return behavioural_data

def combine_behavioural_and_openface(df1 = None, df2 = None):
    combined_data = pd.concat([df1, df2], axis = 1)
    return combined_data

#%%

def select_blocks(complete_data, block_selection):
    if block_selection.shape[0] == 3: 
        all_data = complete_data
    elif block_selection.shape[0] == 2: 
        del_block = np.delete([0, 1, 2], block_selection)
        all_data = complete_data.loc[complete_data["block_count"] != del_block[0]]
    else: 
        all_data = complete_data.loc[complete_data["block_count"] == block_selection[0]]
    return all_data

def select_columns(all_data = None, fix_cols = None, cols_of_interest = None): 
    columns = np.concatenate([fix_cols, cols_of_interest])
    data_selection = all_data.loc[:, columns]
    return data_selection












    
