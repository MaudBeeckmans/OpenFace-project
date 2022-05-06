# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:33:06 2021

@author: Maud
"""


import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from statsmodels.stats import multitest
from Functions import import_data, delete_unsuccessful, delete_incorrect_last2blocks, delete_participant
from Functions import delete_pp_block, display_scores, end_scores

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


#%%
"""Select which subset of the data you'd like to use"""
participants = np.unique(Successful_data2['pp_number']).astype(int)
blocks = np.array([0, 1, 2])
formats = ['--o', '--x', '--v']
colors = ['red', 'green', 'orange']
frames = np.arange(0, 60, 1)

store_all_means = np.empty([participants.shape[0], blocks.shape[0], frames.shape[0]])
store_all_std = np.empty([participants.shape[0], blocks.shape[0], frames.shape[0]])

scaler = StandardScaler()
ordinal_encoder = OrdinalEncoder()

for pp in participants: 
    print("we're at pp {}".format(pp))
    # fig, axs = plt.subplots(1, 1)
    # axs.set_ylim(0, 1.05)
    
    sub_data_df = Successful_data.loc[Successful_data["pp_number"] == pp]
    for block_select in blocks: 
    
        data_df = sub_data_df.loc[sub_data_df["block_count"] == block_select] # 60 frames per trial, 150 trials per block: 9000 rows
        
        #Do the classification
        
        """Put the conditions to ordinal values"""
        Cond_cat = data_df[['Affect']]
        data_df_cond_encoded = ordinal_encoder.fit_transform(Cond_cat)
        data_df.insert(2, "Cond_binary", data_df_cond_encoded, True)
        
        all_mean = []
        all_std = []
        
        for this_frame in np.unique(data_df['frame']):
            """First step: do classification on 1 frame only: here the frame is put to 20"""
            data_f = data_df.loc[data_df["frame"] ==this_frame]
            data_f = data_f.reset_index()
            
            
            """create the x and y variables for strat_train_set & strat_test_set"""
            AU_col = [col for col in data_f.columns if ('AU' in col and '_r' in col)] 
            x, y = data_f[AU_col], data_f['Cond_binary']
            #transform the y-data to integers, as this is often required by ML algorithms
            y = y.astype(np.uint8)
            
            """Feature-scaling: might have to do this (see book page 109)
            - important: fit the scalers to training set only
            - is this correctly done? """
            """I don't think scaling is necessary as all AUs have the same scale? 
            Maybe they should be mean-centered though?"""
            # scaler.fit(train_x)
            # train_x_standardized = pd.DataFrame.from_records(scaler.transform(train_x))
            # train_x_standardized.columns = train_x.columns
            # test_x_standardized = pd.DataFrame.from_records(scaler.transform(test_x))
            # test_x_standardized.columns = test_x.columns
            
            
            """The actual classfication"""
            from sklearn import svm
            classifier = svm.SVC(kernel = 'linear', C = 1)
            
            
            """work with k-fold cross-validation"""
            # print("\nNow doing k-fold cross-validation")
            # cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 1473)
            
            # cross_scores = cross_val_score(classifier, train_x_standardized, train_y, cv=10, scoring="accuracy")
            cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", n_jobs = -1)
            # cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="roc_auc", n_jobs = -1)
            
            # display_scores(cross_scores)
            mean, std = end_scores(cross_scores)
            
            all_mean.append(mean)
            all_std.append(std)
        store_all_means[pp-1, block_select, :] = all_mean
        store_all_std[pp-1, block_select, :] = all_std
        #Plot the classification outcome
        # plt.errorbar(frames, all_mean, yerr = all_std, fmt = formats[block_select], color = colors[block_select], label = 'block {}'.format(block_select))
        
    # axs.plot(frames, np.repeat(0.5, len(frames)), color = 'red', label = 'Chance level')
    # axs.plot([15,15],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='appeared')
    # axs.plot([46,46],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right")
    # fig.suptitle('Classification for pp {}'.format(pp))
    # fig.savefig('FperF_allAUs_pp{}.png'.format(pp+1))
#%%
"""Figure to show the average classification over participants"""

correction = 'holm' # should be holm or fdr 

frames_of_interest = np.arange(0, 60, 1)

fig, axs = plt.subplots(1, 1)
axs.set_ylim(0.35, 1.05)
# axs.set_ylim(0, 0.7)
significant_frames = np.array([])
p_values = np.empty((3, 60))
blocks = np.array([0, 1, 2])
# blocks = [0, 1]
for block in blocks: 
    means = np.mean(store_all_means[:, block, :], axis = 0)
    stds = np.std(store_all_means[:, block, :], axis = 0)
    plt.errorbar(frames, means, yerr = stds, fmt = formats[block], color = colors[block], label = 'block {}'.format(block+1))
    for frame in frames: 
        statistic, p_value = wilcoxon(store_all_means[:, block, frame] - 0.50, alternative = 'greater')
        p_values[block, frame] = p_value

    if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[block, frames_of_interest], alpha=0.05, method='indep', is_sorted=False)
    elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[block, frames_of_interest], alpha = 0.05, method = 'holm')
    if np.any(signif_frames == True): axs.plot(frames_of_interest[signif_frames], np.repeat(0.45-block*0.025, frames_of_interest[signif_frames].shape[0]), 'o', color = colors[block], markersize = 5)
    
       
axs.plot(frames, np.repeat(0.5, len(frames)), color = 'black', label = 'Chance level')
axs.plot([15,15],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='appeared')
axs.plot([46,46],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='disappeared')
handles, labels = axs.get_legend_handles_labels()
handles = np.delete(handles, [1, 2])
labels = np.delete(labels, [1, 2])
fig.legend(handles, labels, loc="center right")
fig.suptitle('Classification scores averaged over all pp')
axs.set_title("Correction method: {}, frames included: {}-{}".format(correction, 
                                                                     frames_of_interest[0]+1, 
                                                                     frames_of_interest[-1]+1))
fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'F_per_F_{}_frames{}to{}_below85deleted{}.png'.format(correction, 
                                                                          frames_of_interest[0]+1, 
                                                                          frames_of_interest[-1]+1, delete_below85)))

#%%

"""Check whether the splitting procedure yields correct splitting values"""

for train_ix, test_ix in cv.split(x, y):
	# select rows
	train_X, test_X = x.iloc[train_ix, :], x.iloc[test_ix, :]
	train_y, test_y = y[train_ix], y[test_ix]
	# summarize train and test composition
	train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

        
        




