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
from Functions import import_data, delete_unsuccessful, delete_incorrect_last2blocks, delete_participant
from Functions import delete_pp_block, display_scores, end_scores
from statsmodels.stats import multitest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from sklearn import svm

delete_below85 = True


openface_map = r"C:\Users\maudb\Documents\Psychologie\2e_master_psychologie\Master_thesis\Pilot_Master_thesis\OpenFace_output"
all_data = import_data(pp_numbers = np.array([["1", "10"],["11", "20"], ["21", "34"]]), datafile_path = openface_map)
accurate_data = delete_incorrect_last2blocks(data = all_data)
Successful_data = delete_unsuccessful(data = accurate_data)
# Successful_data = all_data[all_data["success"] == 1]

Successful_data["pp_number"] = np.where(Successful_data["pp_number"] == 317, 17, 
                                        Successful_data["pp_number"])

Successful_data2 = delete_participant(Successful_data, pp_to_delete = 34)

if delete_below85 == True: 
    # delete pp. 28, block 2 and pp. 30 block 2 (accuracy < 85%)
    Successful_data2 = delete_pp_block(Successful_data2, 28, 1) # block number as stored (2nd block thus deleted)
    Successful_data2 = delete_pp_block(Successful_data2, 30, 1)


#%% Poging: gemiddelde over frame 20-45 gebruiken voor classificatie 

k_fold = 10

scaler = StandardScaler()
ordinal_encoder = OrdinalEncoder()

"""Select which subset of the data you'd like to use"""
participants = np.unique(Successful_data2['pp_number']).astype(int)
blocks = np.array([0, 1, 2])

# frames = np.arange(20, 46, 1)

frames_before_stimulus = np.arange(0, 15, 1)
frames_during_stimulus = np.arange(15, 45, 1)
frames_after_stimulus = np.arange(45, 60, 1)

included_frames = [frames_before_stimulus, frames_during_stimulus, frames_after_stimulus]


store_all_means = np.empty([participants.shape[0], blocks.shape[0]])
store_all_std = np.empty([participants.shape[0], blocks.shape[0]])

AU_col = [col for col in all_data.columns if ('AU' in col and '_r' in col)] 
fixed_cols = ['pp_number', 'block_count', 'Frame_count', 'Trial_number', 'Affect']


store_all_means = np.empty([participants.shape[0], blocks.shape[0], len(included_frames)])
store_all_std = np.empty([participants.shape[0], blocks.shape[0], len(included_frames)])


for pp in participants: 
    print("we're at pp {}".format(pp))
    
    pp_data = Successful_data.loc[Successful_data["pp_number"] == pp]
    # print("PP data: {}".format(np.any(pp_data.isna())))
    test_data = pp_data[np.concatenate([fixed_cols, AU_col])]
    
    Cond_cat = test_data[['Affect']]
    test_data_encoded = ordinal_encoder.fit_transform(Cond_cat)
    test_data.insert(2, "Cond_binary", test_data_encoded, True)
    # print("Test data: {}".format(np.any(test_data.isna())))
    "Creëer nieuwe dataset: voor elke trial slechts 1 value: gemiddelden AU"
    
    
    #mean AU activation over all frames, but have to do this per trial!!
    block_count = 0
    for block in blocks:
        block_data = test_data[test_data["block_count"] == block]
        # print("Block data: {}".format(np.any(block_data.isna())))
        unique_values, indices = np.unique(block_data['Trial_number'], return_index = True)
        shorter_data = block_data.iloc[indices, :].copy(deep=False)
        # print("Shorter data: {}".format(np.any(shorter_data.isna())))
        i = 0
        for frames_of_interest in included_frames: 
            
            relevant_frames_data = block_data[block_data['Frame_count'].isin(frames_of_interest)]
            # print("Frames data: {}".format(np.any(relevant_frames_data.isna())))
            
            """Probleem met de nan values zit bij de mean AU data"""
            # meanAU_data = np.array([np.mean(relevant_frames_data[relevant_frames_data['Trial_number'] == trial][AU_col].to_numpy(), 0) for trial in unique_values])
            meanAU_data = np.array([relevant_frames_data[relevant_frames_data['Trial_number'] == trial][AU_col].mean() for trial in unique_values])
            meanAU_df = pd.DataFrame(meanAU_data, columns = AU_col)
            # print("mean AU data: {}".format(np.any(meanAU_df.isna())))

            
            shorter_data.iloc[:, 6::] = meanAU_data[:, :]
            # print(np.sum(np.sum(shorter_data.isna(), axis = 1) != 0))
            
            cleaned_data = shorter_data.dropna()
            # print(np.any(cleaned_data.isna()))
            """Do the classification"""
            x, y = cleaned_data[AU_col], cleaned_data['Cond_binary']
            #transform the y-data to integers, as this is often required by ML algorithms
            y = y.astype(np.uint8)
            
            
            """The actual classfication"""
            classifier = svm.SVC(kernel = 'linear', C = 1)
            # cv = StratifiedKFold(n_splits=5, random_state=75472, shuffle=True)
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
            """work with k-fold cross-validation"""
            cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", n_jobs = -1)
            # cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", error_score='raise')
            # display_scores(cross_scores)
            mean, std = end_scores(cross_scores)
            
            store_all_means[pp-1, block, i] = mean # ik gebruik gemiddeldes van elke participant, misschien median gebruken? 
            store_all_std[pp-1, block , i] = std #gebruik ik niet 
            
            i += 1
#%%
rep = 3
correction = 'fdr' # should be holm or fdr or bonferroni

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
    ax.set_title("Block  {}".format(block+1))
    ax.axhline(y = 0.5, label = "Chance level", color = 'k')    
    for frame_subset in range(len(included_frames)): 
        means = np.nanmean(store_all_means[:, block, frame_subset], axis = 0)
        stds = np.nanstd(store_all_means[:, block, frame_subset], axis = 0)
        ax.errorbar(frame_subset, means, yerr = stds, fmt = formats[block], color = colors[block], label = '')
    
        statistic, p_value = wilcoxon(store_all_means[:, block, frame_subset] - 0.50, alternative = 'greater')
        """Problem: p-values do take nan into account I think!"""
        p_values[block, frame_subset] = p_value
    if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[block, :], alpha=0.05, method='indep', is_sorted=False)
    elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[block, :], alpha = 0.05, method = 'holm')
    elif correction == 'bonferroni': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[block, :], alpha = 0.05, method = 'bonferroni')
    if np.any(signif_frames == True): 
        ax.plot(np.arange(0, 3, 1)[signif_frames], np.repeat(1, np.sum(signif_frames)), '*', color = 'black', markersize = 5)

fig.suptitle("Classification scores averaged over all pp \nCorrection method: {}"
             .format(correction))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right")
fig.tight_layout()

fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'AverageAU_{}_below85deleted{}_{}fold_rep{}.png'
                         .format(correction, delete_below85, k_fold, rep)))




