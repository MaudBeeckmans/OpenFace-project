# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:57:00 2022

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
#%% Analysis for FperF

frame_selection, frameselection_names, n_subsets = select_frames(analysis_type = 'FperF', data = cleaned_data)
participants = np.unique(cleaned_data.pp_number).astype(int)
blocks = np.unique(cleaned_data.block_count).astype(int)

#Create empty arrays to store all the obtained mean accuracies for each pp. in each block at each frame_selection and within each repetition
store_all_means = np.empty([participants.shape[0], blocks.shape[0]-1, n_subsets])

for ipp, pp in zip(participants-1, participants): 
    print("we're at pp {}".format(pp))
    #select the data relevant for this participant 
    pp_data = cleaned_data.loc[cleaned_data["pp_number"] == pp]
    
    for subset_frame, isubset in zip(frame_selection, range(n_subsets)): 
        subset_data = pp_data.loc[np.isin(pp_data.Frame_count, subset_frame)]
        
        train_data = subset_data[subset_data.block_count == 2]
        train_x, train_y = train_data[AU_cols], train_data['Cond_binary']
        classifier = svm.SVC(kernel = 'linear', C = 1)
        classifier.fit(train_x, train_y)
          
        for iblock, block in zip(blocks[:2], blocks[:2]+1): 
            if (pp == 28 and block == 2) or (pp == 30 and block == 2): 
                store_all_means[ipp, iblock, :] = np.nan
            else: 
                test_data = subset_data[subset_data.block_count == iblock]
                test_x, test_y = test_data[AU_cols], test_data['Cond_binary']
                accuracy = classifier.score(test_x, test_y)
                
                store_all_means[ipp, iblock, isubset] = accuracy

#Store the results in a file!
array_dir = os.path.join(os.getcwd(), 'Stored_results')
if not os.path.isdir(array_dir): os.makedirs(array_dir)
np.save(os.path.join(array_dir, "mean_accuracies_{}_crossblocks.npy".format('FperF')), store_all_means)

#%%Analysis for meanAU
frame_selection, frameselection_names, n_subsets = select_frames(analysis_type = 'meanAU', data = cleaned_data)
participants = np.unique(cleaned_data.pp_number).astype(int)
blocks = np.unique(cleaned_data.block_count).astype(int)

#Create empty arrays to store all the obtained mean accuracies for each pp. in each block at each frame_selection and within each repetition
store_all_means = np.empty([participants.shape[0], blocks.shape[0]-1, n_subsets])

def takemean_1pp1block1subsetdata(start_data = None): 
    all_included_trials, indices = np.unique(start_data.Trial_number, return_index = True)
    n_trials = all_included_trials.shape[0]
    final_data_template = start_data.iloc[indices, :].copy(deep=False)
    final_data_template = final_data_template.iloc[:, :6]
    final_data_template.index = np.arange(0, final_data_template.shape[0])
    meanAU_pertrial = np.array([start_data[start_data.Trial_number == trial][AU_cols].mean() for trial in all_included_trials])
    meanAU_df = pd.DataFrame(meanAU_pertrial , columns = AU_cols)
    final_data = final_data_template.join(meanAU_df)
    return final_data

for ipp, pp in zip(participants-1, participants): 
    print("we're at pp {}".format(pp))
    #select the data relevant for this participant 
    pp_data = cleaned_data.loc[cleaned_data["pp_number"] == pp]
    
    for subset_frame, isubset in zip(frame_selection, range(n_subsets)): 
        subset_data = pp_data.loc[np.isin(pp_data.Frame_count, subset_frame)]
        train_data = subset_data[subset_data.block_count == 2]
        
        #take mean within trial for the train data 
        train_data_averaged = takemean_1pp1block1subsetdata(start_data = train_data)
        train_x, train_y = train_data_averaged[AU_cols], train_data_averaged['Cond_binary']
        classifier = svm.SVC(kernel = 'linear', C = 1)
        classifier.fit(train_x, train_y)
        
        for iblock, block in zip(blocks[:2], blocks[:2]+1): 
            if (pp == 28 and block == 2) or (pp == 30 and block == 2): 
                store_all_means[ipp, iblock, :] = np.nan
            else: 
                test_data = subset_data[subset_data.block_count == iblock]
                test_data_averaged = takemean_1pp1block1subsetdata(start_data = test_data)
                test_x, test_y = test_data_averaged[AU_cols], test_data_averaged['Cond_binary']
                accuracy = classifier.score(test_x, test_y)
                
                store_all_means[ipp, iblock, isubset] = accuracy

#Store the results in a file!
array_dir = os.path.join(os.getcwd(), 'Stored_results')
if not os.path.isdir(array_dir): os.makedirs(array_dir)
np.save(os.path.join(array_dir, "mean_accuracies_{}_crossblocks.npy".format('meanAU')), store_all_means)
        

