# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:45:29 2022

@author: maudb
"""

import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from Functions import import_data, select_columns, select_blocks, delete_unsuccessful, delete_incorrect_last2blocks, delete_participant

delete_below85 = True 

openface_map = r"C:\Users\maudb\Documents\Psychologie\2e_master_psychologie\Master_thesis\Pilot_Master_thesis\OpenFace_output"
all_data = import_data(pp_numbers = np.array([["1", "10"],["11", "20"], ["21", "34"]]), datafile_path = openface_map)
accurate_data = delete_incorrect_last2blocks(data = all_data)
Successful_data = delete_unsuccessful(data = accurate_data)
# Successful_data = all_data[all_data["success"] == 1]

Successful_data["pp_number"] = np.where(Successful_data["pp_number"] == 317, 17, 
                                        Successful_data["pp_number"])

#Delete pp 34: she did not understand the task
Successful_data2 = delete_participant(Successful_data, pp_to_delete = 34)

if delete_below85 == True: 
    # delete pp. 28, block 2 and pp. 30 block 2 (accuracy < 85%)
    Successful_data2 = delete_pp_block(Successful_data2, 28, 1) # block number as stored (2nd block thus deleted)
    Successful_data2 = delete_pp_block(Successful_data2, 30, 1)

#%%
# Should fill in blocks = 0, 1 and 2 once 
blocks = np.array([0]) # block(s) for which you want to plot. - blocks range is 0-2

output_dir = os.path.join(os.getcwd(), 'AU_activationplots', 'Block{}'.format(blocks[0]+1))
if not os.path.isdir(output_dir): os.makedirs(output_dir)

"""Define the data that will be used"""
fixed_cols = ['pp_number', 'block_count', 'frame', 'Affect']
test_cols = [col for col in Successful_data2.columns if ('AU' in col and '_r' in col)]
nrows_group = 3
ncols_group = 6

selected_data = select_columns(all_data = Successful_data2, fix_cols = fixed_cols, cols_of_interest = test_cols)
block_data = select_blocks(selected_data, blocks)

"""Define some relevant variables"""
frames = np.unique(block_data.frame)
n_frames = frames.shape[0]
participants = np.unique(block_data['pp_number']).astype(int)
n_pp = participants.shape[0]
condition_colors = ['-b', '-r']
conditions = ['positive', 'negative']
n_conditions = len(conditions)

line_coordinates = [-1, 1]

#%% Create plot with mean activation per participant over frames for each AU 
n_rows = nrows_group
n_cols = ncols_group

for col in test_cols:
    # Prepare the figure     
    fig, axes = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = True)
    fig.suptitle('{} activation positive minus negative \n Block {}'.format(col, blocks[0]+1))
    
    #Store per participant the mean activation in + and - per frame
    store_activations = np.empty([n_pp, n_frames, n_conditions])
    
    for cond_count in range(n_conditions):
        axes.set_xlabel('frame', loc = 'right')
        axes.set_ylabel('difference', loc = 'top')
        condition_data = block_data.loc[block_data["Affect"] == conditions[cond_count]]
        
        for pp in participants: 
            pp_data = condition_data.loc[condition_data["pp_number"] == pp]
            mean_framedata = np.array([np.mean(pp_data.loc[pp_data["frame"] == frame][col]) for frame in frames])
            store_activations[pp-1, :, cond_count] = mean_framedata
    
    #Define whether there is a significant difference between the 2 conditions at each frame over all participants
    P_values = np.array([stats.ttest_rel(store_activations[:, frame-1, 0], store_activations[:, frame-1, 1])[-1] for frame in frames])
    
    #Correct P_values
    significant_frames, corrected_P_values = multitest.fdrcorrection(P_values, alpha=0.05)
    
    #Plot the difference in AU activation for each participant over the frames
    [plt.plot(frames, store_activations[pp-1, :, 0] - store_activations[pp-1, :, 1], alpha = 0.3) for pp in participants]
    #Plot the mean difference in AU activation over all participants
    group_activations = np.mean(store_activations[:, :, 0], axis = 0) - np.mean(store_activations[:, :, 1], axis = 0) 
    plt.plot(frames, group_activations, label = "group average", color = 'green')
    #Plot the significance of the frames
    if np.any(significant_frames == True): 
        ymin, ymax = axes.get_ylim()
        axes.plot(frames[significant_frames], np.repeat(ymin + 0.05, frames[significant_frames].shape[0]), 'o', color = 'green', markersize = 5)
    # fig.savefig(os.path.join(output_dir, "{}_block{}".format(col, blocks[0]+1)))
    
            
            
            



