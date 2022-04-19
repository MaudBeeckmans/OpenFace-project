# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:17:44 2022

@author: maudb
"""


import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from Functions import import_data, import_behaviouraldata, combine_behavioural_and_openface, select_columns, select_blocks

openface_map = r"C:\Users\maudb\Documents\Psychologie\2e master psychologie\Master thesis\Pilot_Master_thesis\OpenFace output"
#Combined output: contains both the behavioural output as well as the openface processed output!
Combined_output = import_data(datafile_path=openface_map)
Combined_columns = Combined_output.columns
# SHAPE of combined_output: 3 blocks * 150trials per block * 60 frames per trial * 10 participants



# %% Check accuracy in each block 
Trial_data = Combined_output.loc[Combined_output["Frame_count"] == 0]

"""Accuracy over all participants"""

#!: accuarcy, -1 = no response, 0 = wrong response, 1 = correct response
print("Calculate the accuracy for each block")
blocks = np.arange(0, 3, 1)
for block in blocks:
    Block_Trialdata = Trial_data.loc[Trial_data["block_count"] == block]
    Mean_accuracy = np.mean(Block_Trialdata['accuracy'] > 0)
    print("Mean accuracy for block {} is {}".format(block, np.round(Mean_accuracy, 2)))
    

"""Accuracy for each participant separately"""
#Check whether accuracy was high enough for each participant in each block
participants = np.arange(1, 11, 1)
for block in blocks: 
    Block_Trialdata = Trial_data.loc[Trial_data["block_count"] == block]
    for pp in participants: 
        Block_pp_Trialdata = Block_Trialdata.loc[Block_Trialdata['pp_number'] == pp]
        mean_accuracy = np.mean(Block_pp_Trialdata['accuracy'] > 0)
        if mean_accuracy < 0.85: print("Participant {} did not achieve high enough accuracy in block {}".format(pp, block))        

#%% Check success of OpenFace processing at all frames 
total_frames = Combined_output.shape[0]
n_failed_frames = np.sum(Combined_output['success'] == 0)
percentage_failed_frames = n_failed_frames / total_frames * 100
print("Percentage of failed OpenFace processed frames: {}%".format(np.round(percentage_failed_frames, 3)))
Succesful_output = Combined_output[Combined_output["success"] == 1]
Failed_output = Combined_output[Combined_output["success"] == 0]

"""Failed frames not unique to any block, participant or frame"""
print("participants included in the failed data: {}".format(np.unique(Failed_output["pp_number"])))
print("blcoks included in the failed data: {}".format(np.unique(Failed_output["block_count"])))
print("frames included in the failed data: {}".format(np.unique(Failed_output["Frame_count"])))







# fixed_cols = ['pp_number', 'block_count', 'frame', 'Affect']
# test_cols = [col for col in Combined_columns if ('AU' in col and '_r' in col)]



# block = np.array([2])

# #Select the relevant data for ALL participants
# selected_data = select_columns(all_data = Combined_output, fix_cols = fixed_cols, cols_of_interest = test_cols)
# block_data = select_blocks(selected_data, block)

# block_data_pos = block_data.loc[block_data["Affect"] =="positive"]
# block_data_neg = block_data.loc[block_data["Affect"] =="negative"]
# conditions = [block_data_pos, block_data_neg]


        
















# # %%
# behavioural_map = r"C:\Users\maudb\Documents\Psychologie\2e master psychologie\Master thesis\Pilot_Master_thesis\Ruwe video's"
# Behavioural_output = import_behaviouraldata(datafile_path=behavioural_map)
# Behavioural_columns = Behavioural_output.columns


# Combine behavioural and openface output
# combined_data = combine_behavioural_and_openface(df1 = Behavioural_output, df2 = Openface_output)