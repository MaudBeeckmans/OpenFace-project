# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:14:15 2022

@author: maudb
"""

import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from Functions import import_data

openface_map = r"C:\Users\maudb\Documents\Psychologie\2e master psychologie\Master thesis\Pilot_Master_thesis\OpenFace output"
all_data = import_data(datafile_path = openface_map)
Successful_data = all_data[all_data["success"] == 1]


"""Create some functions to use in the loop"""
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
        
def end_scores(scores): 
    mean = scores.mean()
    std = scores.std()
    return mean, std 

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.stats import wilcoxon


#%% Poging: gemiddelde over frame 20-45 gebruiken voor classificatie 

"""Select which subset of the data you'd like to use"""
participants = np.arange(0, 10, 1)
blocks = np.array([0, 1, 2])

# frames = np.arange(20, 46, 1)

frames_before_stimulus = np.arange(0, 15, 1)
frames_during_stimulus = np.arange(15, 45, 1)
frames_after_stimulus = np.arange(45, 60, 1)


store_all_means = np.empty([participants.shape[0], blocks.shape[0]])
store_all_std = np.empty([participants.shape[0], blocks.shape[0]])

AU_col = [col for col in all_data.columns if ('AU' in col and '_r' in col)] 
fixed_cols = ['pp_number', 'block_count', 'Frame_count', 'Trial_number', 'Affect']


store_all_means = np.empty([participants.shape[0], blocks.shape[0], 3])
store_all_std = np.empty([participants.shape[0], blocks.shape[0], 3])




for pp in participants: 
    print("we're at pp {}".format(pp))
    # fig, axs = plt.subplots(1, 1)
    # axs.set_ylim(0, 1.05)
    
    pp_data = Successful_data.loc[Successful_data["pp_number"] == pp+1]
    test_data = pp_data[np.concatenate([fixed_cols, AU_col])]
    
    Cond_cat = test_data[['Affect']]
    test_data_encoded = ordinal_encoder.fit_transform(Cond_cat)
    print(np.unique(ordinal_encoder.categories_))
    test_data.insert(2, "Cond_binary", test_data_encoded, True)
    
    "Creëer nieuwe dataset: voor elke trial slechts 1 value: gemiddelden AU"
    
    
    #mean AU activation over all frames, but have to do this per trial!!
    for block_select in blocks:
        block_data = test_data[test_data["block_count"] == block_select]
        unique_values, indices = np.unique(block_data['Trial_number'], return_index = True)
        shorter_data = block_data.iloc[indices, :]
        i = 0
        for frames_of_interest in [frames_before_stimulus, frames_during_stimulus, frames_after_stimulus]: 
            
            relevant_frames_data = block_data[block_data['Frame_count'].isin(frames_of_interest)]
            
            meanAU_data = np.array([np.mean(relevant_frames_data[relevant_frames_data['Trial_number'] == trial][AU_col], 0) for trial in unique_values])
            
            
            shorter_data.loc[:, AU_col] = meanAU_data
            
            """Do the classification"""
            x, y = shorter_data[AU_col], shorter_data['Cond_binary']
            #transform the y-data to integers, as this is often required by ML algorithms
            y = y.astype(np.uint8)
                
            """The actual classfication"""
            from sklearn import svm
            classifier = svm.SVC(kernel = 'linear', C = 1)
            cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
                
            """work with k-fold cross-validation"""
            print("\nNow doing k-fold cross-validation")
            cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", n_jobs = -1)
            
            display_scores(cross_scores)
            mean, std = end_scores(cross_scores)
            
            store_all_means[pp, block_select, i] = mean
            store_all_std[pp, block_select, i] = std
            
            i += 1
        #Plot the classification outcome
        # plt.errorbar(20, mean, yerr = std, fmt = formats[block_select], color = colors[block_select], label = 'block {}'.format(block_select))
        
    # axs.plot(frames, np.repeat(0.5, len(frames)), color = 'red', label = 'Chance level')
    # axs.plot([15,15],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='appeared')
    # axs.plot([46,46],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right")
    # fig.suptitle('Classification for pp {}'.format(pp+1))
    # fig.savefig('FperF_allAUs_pp{}.png'.format(pp+1))
#%%

formats = ['--o', '--o', '--o']
colors = ['red', 'green', 'orange']

"""Figure to show the average classification over participants"""
fig, axs = plt.subplots(1, 3, sharex = True, sharey = True)
axs[0].set_ylim(0, 1.05)
p_values = np.empty((3, 3))
labels = ['F 0-15', 'F 15-45', 'F 45-60']
plt.xticks(np.array([0, 1, 2]), labels)
for block in blocks: 
    ax = axs[block]
    ax.set_title("Block  {}".format(block))
    for frame_subset in range(3): 
        means = np.nanmean(store_all_means[:, block, frame_subset], axis = 0)
        stds = np.nanstd(store_all_means[:, block, frame_subset], axis = 0)
        ax.errorbar(frame_subset, means, yerr = stds, fmt = formats[block], color = colors[block], label = '')
    
        statistic, p_value = wilcoxon(store_all_means[:, block, frame_subset] - 0.50)
        p_values[block, frame_subset] = p_value
        if p_value <= 0.05: ax.plot(frame_subset, 1, '*', color = 'grey', linewidth = 0.1)
    
fig.suptitle('Classification scores averaged over all pp')
fig.tight_layout()
# fig.savefig('F_per_F_averageaccuracyoverallpp.png')
print(p_values)
# [0.85895492 0.01367188 0.00195312]


# ==> Idea: boxplots instead of mean & std plotting? 






















