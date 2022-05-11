# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:19:40 2022

@author: maudb
"""

#%%Import all relevant modules 
import pyreadr, os, random
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from statsmodels.stats import multitest
from Functions import import_data, delete_unsuccessful, delete_incorrect_last2blocks, delete_participant
from Functions import delete_pp_block, display_scores, end_scores, select_columns, select_frames

random.seed(25)
scaler = StandardScaler()
ordinal_encoder = OrdinalEncoder()

#%%Load all the data & delete unsuccesful & inaccurate trials 
delete_below85 = True

openface_map = r"C:\Users\maudb\Documents\Psychologie\2e_master_psychologie\Master_thesis\Pilot_Master_thesis\OpenFace_output"
all_data = import_data(pp_numbers = np.array([["1", "10"],["11", "20"], ["21", "34"]]), datafile_path = openface_map)

accurate_data = delete_incorrect_last2blocks(data = all_data)
Successful_data = delete_unsuccessful(data = accurate_data)

#%%Reduce size of the dataset & delete participants whose accuracy was below 85 and who did not understand task

#Reduce size of the dataset 
AU_cols = [col for col in all_data.columns if ('AU' in col and '_r' in col)] 
fixed_cols = ['pp_number', 'block_count', 'Frame_count', 'Trial_number', 'Affect']
smaller_data = select_columns(all_data = Successful_data, fix_cols = fixed_cols, cols_of_interest = AU_cols)

#change pp number of pp. 317 to 17
smaller_data["pp_number"] = np.where(smaller_data.pp_number == 317, 17, smaller_data.pp_number)

#Delete pp 34: she did not understand the task
cleaned_data = delete_participant(smaller_data, pp_to_delete = 34)
#Delete block 1 for participant 28 and 30: their accuracy was below 85% in this block 
if delete_below85 == True: 
    # delete pp. 28, block 2 and pp. 30 block 2 (accuracy < 85%)
    cleaned_data = delete_pp_block(cleaned_data, 28, 1) # block number as stored (2nd block thus deleted)
    cleaned_data = delete_pp_block(cleaned_data, 30, 1)

# Convert conditions to binary conditions
Conditions_data = cleaned_data[['Affect']]
Conditions_data_encoded = ordinal_encoder.fit_transform(Conditions_data)
cleaned_data.insert(2, "Cond_binary", Conditions_data_encoded, True)


#start_data should be block_data
def take_mean(start_data = None, analysis_type = 'FperF'): 
    if analysis_type == 'FperF': final_data = start_data
    elif analysis_type == 'meanAU': 
        all_included_trials, indices = np.unique(start_data.Trial_number, return_index = True)
        n_trials = all_included_trials.shape[0]
        final_data_template = start_data.iloc[indices, :].copy(deep=False) 
        final_data = final_data_template.append(final_data_template).append(final_data_template)
        final_data.index = np.arange(0, final_data.shape[0], 1)
        for frames_subset, subset_name, subset_count in zip(frame_selection, frameselection_names, range(len(frameselection_names))): 
            
            framessubset_data = start_data[start_data.Frame_count.astype(int).isin(frames_subset)]
            meanAU_framessubset = np.array([framessubset_data[framessubset_data.Trial_number == trial][AU_cols].mean() for trial in all_included_trials])
            meanAU_df = pd.DataFrame(meanAU_framessubset , columns = AU_cols)
            final_data.iloc[n_trials*subset_count:n_trials*(subset_count+1), 6::] = meanAU_df
            final_data.iloc[n_trials*subset_count:n_trials*(subset_count+1), 3] = np.repeat(subset_name, all_included_trials.shape[0])
        final_data = final_data.dropna()
    return final_data

#%%
analysis = 'FperF' # analysis_type should be 'FperF' or 'meanAU'

# frame selection: array when FperF; list when meanAU    
frame_selection, frameselection_names, n_subsets = select_frames(analysis_type = analysis, data = cleaned_data)
participants = np.unique(cleaned_data.pp_number).astype(int)
blocks = np.unique(cleaned_data.block_count).astype(int)

#Create empty arrays to store all the obtained mean accuracies for each pp. in each block at each frame_selection and within each repetition
store_all_means = np.empty([participants.shape[0], blocks.shape[0]-1, n_subsets])


for ipp, pp in zip(participants-1, participants): 
    print("we're at pp {}".format(pp))
    #select the data relevant for this participant 
    pp_data = cleaned_data.loc[cleaned_data["pp_number"] == pp]
    #iblock: starts from 0, block: starts from 1
    
    for selected_frames, subset_names in zip(frame_selection, frameselection_names): 
        
        subset_data = pp_data.loc[pp_data.Frame_count == selected_frames]
        subset_data = subset_data.reset_index()
        
        train_data = subset_data[subset_data.block_count == 2]
        train_data = take_mean(start_data = train_data, analysis_type = analysis)
        train_x, train_y = train_data[AU_cols], train_data['Cond_binary']
        classifier = svm.SVC(kernel = 'linear', C = 1)
        classifier.fit(train_x, train_y)
        # if selected_frames == 1: 
        #     print(np.unique(train_data.Frame_count))
        #     print(np.unique(subset_data.Frame_count))
        #     print(np.unique(subset_data.block_count))
        
        for iblock, block in zip(blocks[:2], blocks[:2]+1): 
            if (pp == 28 and block == 2) or (pp == 30 and block == 2): 
                store_all_means[ipp, iblock, :] = np.nan
            else: 
                block_data = subset_data[subset_data.block_count == iblock]
                test_data = take_mean(start_data = block_data, analysis_type = analysis)
                
                test_x, test_y = test_data[AU_cols], test_data['Cond_binary']
                accuracy = classifier.score(test_x, test_y)
                
                store_all_means[ipp, iblock, selected_frames] = accuracy

#Store the results in a file!
array_dir = os.path.join(os.getcwd(), 'Stored_results')
if not os.path.isdir(array_dir): os.makedirs(array_dir)
np.save(os.path.join(array_dir, "mean_accuracies_{}_crossblocks.npy".format(analysis)), store_all_means)
