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

#%% Functions to clean the data
def delete_unsuccessful(data): 
    Successful_data = data[data["success"] == 1]
    return Successful_data

def delete_participant(data, pp_to_delete): 
    Cleaned_data = data[data["pp_number"] != pp_to_delete]
    return Cleaned_data

def delete_incorrect_last2blocks(data): 
    Cleaned_data = data.loc[data["block_count"] == 0]
    for block in [1, 2]: 
        Block_data = data.loc[data["block_count"] == block]
        Cleaned_Block_data = Block_data.drop(Block_data[Block_data.accuracy <= 0].index)
        # print(np.all(Cleaned_Block_data.accuracy == 1))    
        Cleaned_data = pd.concat([Cleaned_data, Cleaned_Block_data])
    return Cleaned_data

def delete_pp_block(data, pp, block): 
    Other_blocks = data.loc[data["block_count"] != block]
    Block_to_delete = data.loc[data["block_count"] == block]
    PP_block_deleted = Block_to_delete.drop(Block_to_delete[Block_to_delete.pp_number == pp].index)
    Cleaned_data = pd.concat([Other_blocks, PP_block_deleted])
    return Cleaned_data

#%%
"""Create some functions to use in the loop"""
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", np.mean(scores))
    print("Standard deviation:", np.std(scores))
        
def end_scores(scores): 
    mean = np.mean(scores) # misschien mean, want nan values mogen eigenlijk niet he 
    std = np.std(scores)
    return mean, std 
    

def select_frames(analysis_type = 'FperF', data = None): 
    if analysis_type == 'FperF': 
        frame_selection = np.unique(data.Frame_count).astype(int)
        frameselection_names = frame_selection
        n_subsets = frameselection_names.shape[0]
    elif analysis_type == 'meanAU': 
        frame_selection = [np.arange(0, 15, 1), np.arange(15, 45, 1), np.arange(45, 60, 1)]
        frameselection_names = ['F1-15', 'F16-45', 'F46-60']
        n_subsets = len(frameselection_names)
    return frame_selection, frameselection_names, n_subsets

#%%
def balance_train_data(unbalanced_train_data = None): 
    unique_values, n_values = np.unique(unbalanced_train_data.Cond_binary, return_counts = True)
    n_to_delete = n_values[1] - n_values[0]
    zeroorone = np.asarray(np.where(n_values == np.max(n_values)))[0, 0]
    all_largest = np.asarray(np.where(unbalanced_train_data.Cond_binary == zeroorone)).squeeze()
    indices_to_delete = np.random.choice(all_largest, np.abs(n_to_delete))
    indices = unbalanced_train_data.index
    new_indices = np.delete(indices, indices_to_delete)
    balanced_train_data = unbalanced_train_data.loc[new_indices, :]
    return balanced_train_data




    
