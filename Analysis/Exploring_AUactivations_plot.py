# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:31:14 2022

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
from Functions import delete_pp_block
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
blocks = np.array([0, 1, 2])
output_dir = os.path.join(os.getcwd(), 'AU_activationplots', 'all_blocks')
if not os.path.isdir(output_dir): os.makedirs(output_dir)

"""Define the data that will be used"""
fixed_cols = ['pp_number', 'block_count', 'frame', 'Affect']
test_cols = [col for col in Successful_data2.columns if ('AU' in col and '_r' in col)]
nrows_group = 3
ncols_group = 6 

selected_data = select_columns(all_data = Successful_data2, fix_cols = fixed_cols, cols_of_interest = test_cols)


"""Define some relevant variables"""
frames = np.unique(selected_data.frame)
n_frames = frames.shape[0]
participants = np.unique(selected_data.pp_number).astype(int)
n_pp = participants.shape[0]
condition_colors = ['-b', '-r']
conditions = ['positive', 'negative']
n_conditions = len(conditions)
line_coordinates = [-1, 1]
#%%
relevant_frames = frames[15:]
test_cols = ["AU06_r", "AU12_r"]



for col in test_cols:
    # Prepare the figure     
    fig, axes = plt.subplots(nrows = 1, ncols = blocks.shape[0], sharex = True)
    fig.suptitle('{} activation positive minus negative'.format(col))
    axes[0].set_ylabel('difference', loc = 'top')
    axes[0].set_xlabel('frame', loc = 'right')
    for block in blocks: 
        block_data = select_blocks(selected_data, np.array([block]))
        axes[block].set_title('Block {}'.format(block))
        store_activations = np.empty([n_pp, n_frames, n_conditions])
        
        for cond_count in range(n_conditions):
            condition_data = block_data.loc[block_data["Affect"] == conditions[cond_count]]
            
            for pp in participants: 
                pp_data = condition_data.loc[condition_data["pp_number"] == pp]
                mean_framedata = np.array([np.mean(pp_data.loc[pp_data["frame"] == frame][col]) for frame in frames])
                store_activations[pp-1, :, cond_count] = mean_framedata
        
        #Define whether there is a significant difference between the 2 conditions at each frame over all participants
        P_values = np.array([stats.ttest_rel(store_activations[:, frame-1, 0], store_activations[:, frame-1, 1])[-1] for frame in relevant_frames])
        
        #Correct P_values
        significant_frames, corrected_P_values = multitest.fdrcorrection(P_values, alpha=0.05)
        
        #Plot the difference in AU activation for each participant over the frames
        [axes[block].plot(frames, store_activations[pp-1, :, 0] - store_activations[pp-1, :, 1], alpha = 0.3) for pp in participants]
        #Plot the mean difference in AU activation over all participants
        group_activations = np.nanmean(store_activations[:, :, 0], axis = 0) - np.nanmean(store_activations[:, :, 1], axis = 0) 
        axes[block].plot(frames, group_activations, label = "group average", color = 'green')
        #Plot the significance of the frames
        if np.any(significant_frames == True): 
            ymin, ymax = axes[block].get_ylim()
            axes[block].plot(relevant_frames[significant_frames], np.repeat(ymin + 0.05, relevant_frames[significant_frames].shape[0]), 'o', color = 'green', markersize = 5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "{}".format(col)))
            
        










