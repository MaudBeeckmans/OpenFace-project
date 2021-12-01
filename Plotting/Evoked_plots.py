# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:58:59 2021

@author: Maud
"""

import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

def normalize_data(data_df, normalize_cols = None, fix_cols = None):
    participants = np.arange(1, 11, 1)
    normalized_df = pd.DataFrame(columns = data_df.columns)
    for pp in participants: 
        pp_data = data_df.loc[data_df["pp_number"] == pp]
        pp_data_n = pd.DataFrame(columns = normalize_cols)
        for col in normalize_cols: 
            pp_AU_data = pp_data[col]
            pp_data_n[col] = (pp_AU_data - np.mean(pp_AU_data)) / np.std(pp_AU_data)
        if pp == 0: normalized_df = pp_data_n
        else: normalized_df = pd.concat([normalized_df, pp_data_n])
    for col in fix_cols: 
        normalized_df[col] = data_df[col]
    return normalized_df
def select_blocks(notnorm_pos_data, notnorm_neg_data, norm_pos_data, norm_neg_data, block_selection):
    if block_selection.shape[0] == 3: 
        data_pos = notnorm_pos_data
        data_neg = notnorm_neg_data
        data_pos_Z = norm_pos_data
        data_neg_Z = norm_neg_data
    elif block_selection.shape[0] == 2: 
        del_block = np.delete([0, 1, 2], block_selection)
        data_pos = notnorm_pos_data.loc[notnorm_pos_data["block_count"] != del_block[0]]
        data_neg = notnorm_neg_data.loc[notnorm_neg_data["block_count"] != del_block[0]]
        data_pos_Z = norm_pos_data.loc[norm_pos_data["block_count"] != del_block[0]]
        data_neg_Z = norm_neg_data.loc[norm_neg_data["block_count"] != del_block[0]]
    else: 
        data_pos = notnorm_pos_data.loc[notnorm_pos_data["block_count"] == block_selection[0]]
        data_neg = notnorm_neg_data.loc[notnorm_neg_data["block_count"] == block_selection[0]]
        data_pos_Z = norm_pos_data.loc[norm_pos_data["block_count"] == block_selection[0]]
        data_neg_Z = norm_neg_data.loc[norm_neg_data["block_count"] == block_selection[0]]
    return data_pos, data_neg, data_pos_Z, data_neg_Z
def import_data(pp_numbers = np.array([["1", "5"],["6", "10"]]), datafile_path = "C:\\Users\\Maud\\Documents\\Psychologie\\1ste_master_psychologie\\Masterproef\\Final_versions\\OpenFace_processing"): 
    count = 0
    for files in pp_numbers: 
        data_path = os.path.join(datafile_path, str('data_processed_concat' + files[0] + 'to' + 
                                                  files[-1] + '_AUstatic.rds'))
        data = pyreadr.read_r(data_path) # also works for RData
        data_df = data[None]
        if count == 0: all_data_df = data_df
        else: all_data_df = pd.concat([all_data_df, data_df])
        count += 1
    return all_data_df

# import the data (should run this only once)
all_data_df = import_data()

#%% Some stuff that may be adapted 

# this should be adapted to your columns of interest 
test_cols = [col for col in all_data_df.columns if ('p_' in col and 'pp' not in col)] 
nrows_group = 5
ncols_group = 8
variable_of_interest = 'rigid_nonrigid_faceparams'

blocks = np.array([1]) # should be 'all' or a block number 
normalize = True
deg = 60*len(test_cols) #1 when no correction, 60 when frame correction, 60*17 when frame & AU correction
treshold = 0.05/deg

#%% Create the normalized & non-normalized data for each condition
fixed_cols = ['pp_number', 'block_count', 'frame', 'Affect']

test_data = all_data_df[np.concatenate([fixed_cols, test_cols])]
all_test_data_pos = test_data.loc[test_data["Affect"] =="positive"]
all_test_data_neg = test_data.loc[test_data["Affect"] =="negative"]

normalized_test_data = normalize_data(test_data, normalize_cols = test_cols, fix_cols = fixed_cols)
all_test_data_pos_Z = normalized_test_data.loc[normalized_test_data["Affect"] =="positive"]
all_test_data_neg_Z = normalized_test_data.loc[normalized_test_data["Affect"] =="negative"]

del test_data, normalized_test_data, fixed_cols

#%% Select your data of interest

data_pos, data_neg, data_pos_Z, data_neg_Z = select_blocks(all_test_data_pos, all_test_data_neg, 
                                                           all_test_data_pos_Z, all_test_data_neg_Z, blocks)

# print some properties of your current data of interest
print(np.abs(np.mean(data_pos_Z) - np.mean(data_neg_Z)))
print(np.abs(np.std(data_pos_Z) - np.std(data_neg_Z)))

#%% load some general stuff
frames = np.unique(data_pos.frame)

participants = np.arange(0, 10, 1)
condition_colors = ['-b', '-r']
txt = ['positive', 'negative']

if treshold == 0.05: output_folder = os.path.join(os.getcwd(), '{}_Plots'.format(variable_of_interest), 
                                                  'Block{}_normalized{}'.format(blocks, normalize))
else: output_folder = os.path.join(os.getcwd(), '{}_Plots'.format(variable_of_interest), 
                                   'Block{}_normalized{}'.format(blocks, normalize), 
                                   'Bonferroni_corrected_divideby{}'.format(deg))
if not os.path.isdir(output_folder): 
    os.makedirs(output_folder)

if normalize == True: 
    conditions = [data_pos_Z, data_neg_Z]
    line_coordinates = [-1, 1]
else: 
    conditions = [data_pos, data_neg]
    line_coordinates = [0, 2]
    

#%%
for col in test_cols: 
    fig, axes = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True, figsize = [13.5, 7])
    fig.suptitle('Activation for {} in block {}\nTreshold p-value = 0.05/{}'.format(col, blocks, deg))
    axes[1, 4].set_xlabel('frame', loc = 'right')
    axes[0, 0].set_ylabel('activation', loc = 'top')
    for pp in participants: 
        row_select = int(pp/5)
        col_select = pp - (pp >= 5)*1*5
        axes[row_select, col_select].set_title('participant {}'.format(pp+1))
        cond_count = 0
        ttest_basis = np.empty([2, frames.shape[0], 75*blocks.shape[0]])
        for cond in conditions: 
            col_data = np.column_stack([cond['frame'], cond[col], cond['pp_number']])
            pp_data = col_data[col_data[:, 2] == (pp+1), :]
            mean_frames = np.array([])
            for frame in frames: 
                current_framedata = pp_data[(pp_data[:, 0] == frame), 1]
                mean_frame = np.mean(current_framedata)
                mean_frames = np.append(mean_frames, mean_frame)
                ttest_basis[cond_count, frame-1, :] = current_framedata
            
            axes[row_select, col_select].plot(frames, mean_frames, condition_colors[cond_count], label = txt[cond_count])
            cond_count += 1
        significance = np.array([])
        for frame in frames: 
            T_value, P_value = stats.ttest_rel(ttest_basis[0, frame-1, :], ttest_basis[1, frame-1, :])
            significance = np.append(significance, P_value <= treshold)
        significance = (significance==1)
        for plot in frames[significance]:     
            axes[row_select, col_select].plot(plot, 2, 'k.', linewidth = 0.1)
        axes[row_select, col_select].plot([15,15],line_coordinates, lw = 2, linestyle ="dashed", color ='y', label ='appeared')
        axes[row_select, col_select].plot([46,46],line_coordinates, lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels, loc="right")
    fig.savefig(os.path.join(output_folder, '{}_for_allppseparately.png'.format(col)))

#%% Create plot for all columns (1 figure, subplots = test_cols.shape[0])

n_rows = nrows_group
n_cols = ncols_group
fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, sharex = True, sharey = True, figsize = [13.5, 7])
fig.suptitle('Activation averaged over all participants in block {}\nTreshold p-value = 0.05/{}'.format(blocks, deg))
axes[n_rows-1, n_cols-1].set_xlabel('frame', loc = 'right')
axes[0, 0].set_ylabel('activation', loc = 'top')
count = 0
for col in test_cols: 
    row_select = int(count/n_cols)
    col_select = int(count - np.floor(count/n_cols)*n_cols)
    axes[row_select, col_select].plot([15,15],line_coordinates, lw = 2, linestyle ="dashed", color ='y', label ='appeared')
    axes[row_select, col_select].plot([46,46],line_coordinates, lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
    axes[row_select, col_select].set_title('{}'.format(col))
    mean_activation = np.empty([frames.shape[0], 2, participants.shape[0]]) #shape 60x2x10
    cond_count = 0
    for cond in conditions: 
        test_data = np.column_stack([cond['frame'], cond[col], cond['pp_number']])
        for frame in frames: 
            frame_data_Z = test_data[test_data[:, 0] == frame, :]
            for pp in participants: 
                pp_data = frame_data_Z[frame_data_Z[:, 2] == pp+1, :]
                mean_activation[frame-1, cond_count, pp] = np.mean(pp_data[:, 1])
        cond_count += 1
    for i in range(2): 
        mean_cond = np.mean(mean_activation[:, i, :], axis = 1)
        axes[row_select, col_select].plot(frames, mean_cond, condition_colors[i], label = txt[i])
        # below: check whether difference is significant
        significance = np.array([])
    for frame in frames: 
        T_value, P_value = stats.ttest_rel(mean_activation[frame-1, 0, :], mean_activation[frame-1, 1, :])
        significance = np.append(significance, P_value <= treshold)
    significance = (significance==1)
    for plot in frames[significance]:     
        axes[row_select, col_select].plot(plot, 1, 'k.', linewidth = 0.1)
    count += 1

fig.savefig(os.path.join(output_folder, '{}_overallpp.png'.format(variable_of_interest)))
