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
pos_cols = ["AU06_r", "AU12_r"]
neg_cols = ["AU01_r", "AU04_r", "AU12_r"]

test_type = 'negative'
if test_type == 'positive': 
    test_cols = pos_cols
    y_inch = 7
else: 
    test_cols = neg_cols
    y_inch = 9
n_cols = len(test_cols)
fig, axes = plt.subplots(nrows = len(test_cols), ncols = blocks.shape[0], sharex = True)
fig.suptitle('Action units related to {} affect'.format(test_type), fontsize = 20)
# Adding a plot in the figure which will encapsulate all the subplots with axis showing only
fig.add_subplot(1, 1, 1, frame_on=False)

# Hiding the axis ticks and tick labels of the bigger plot
plt.tick_params(labelcolor="none", bottom=False, left=False)

# Adding the x-axis and y-axis labels for the bigger plot
plt.ylabel('Activation difference', loc = 'center', fontsize = 15)
plt.xlabel('Frames', loc = 'right', fontsize = 15)

fig.set_size_inches(16, y_inch)

for col_count, col in zip(range(len(test_cols)), test_cols):
    # axes[col_count, 0].set_ylabel('{}'.format(col[:4]), fontweight = 'bold', loc = 'center')
    
    for block in blocks: 
        block_data = select_blocks(selected_data, np.array([block]))
        axes[col_count, block].set_title('Block {}'.format(block+1), fontsize = 18)
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
        [axes[col_count, block].plot(frames, store_activations[pp-1, :, 0] - store_activations[pp-1, :, 1], alpha = 0.3, linestyle = 'dashed') for pp in participants]
        #Plot the mean difference in AU activation over all participants
        group_activations = np.nanmean(store_activations[:, :, 0], axis = 0) - np.nanmean(store_activations[:, :, 1], axis = 0) 
        axes[col_count, block].plot(frames, group_activations, label = "group average", color = 'green')
        #Plot the significance of the frames
        if np.any(significant_frames == True): 
            ymin, ymax = axes[col_count, block].get_ylim()
            axes[col_count, block].plot(relevant_frames[significant_frames], np.repeat(ymin + 0.05, relevant_frames[significant_frames].shape[0]), 'o', color = 'green', markersize = 5)
    
[axes[col, block].plot([14,14],axes[col, block].get_ylim(), lw = 1, color ='grey', label ='stimulus onset')for col, block in zip(np.repeat(range(n_cols), 3), np.tile(blocks, n_cols))]
[axes[col, block].plot([45,45],axes[col, block].get_ylim(), lw = 1, color ='grey', label ='stimulus offset')for col, block in zip(np.repeat(range(n_cols), 3), np.tile(blocks, n_cols))]
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize = 13)

fig.tight_layout()

fig.savefig(os.path.join(output_dir, "expected_{}".format(test_type)))
            
        










