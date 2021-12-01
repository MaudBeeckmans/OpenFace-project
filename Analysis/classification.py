# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:33:06 2021

@author: Maud
"""

"""To do: adapt the appeared & disappeared
- appeared at frame 15; disappeared at frame 46"""


import pyreadr
import os 
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from Evoked_plots import load_data

all_data_df = load_data()

#%%
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

"""Select which subset of the data you'd like to use"""
participants = np.arange(0, 10, 1)
blocks = np.array([0, 1, 2])
formats = ['--o', '--x', '--v']
colors = ['black', 'blue', 'black']
frames = np.arange(0, 60, 1)

store_all_means = np.empty([participants.shape[0], blocks.shape[0], frames.shape[0]])
store_all_std = np.empty([participants.shape[0], blocks.shape[0], frames.shape[0]])

for pp in participants: 
    print("we're at pp {}".format(pp))
    fig, axs = plt.subplots(1, 1)
    axs.set_ylim(0, 1.05)
    
    sub_data_df = all_data_df.loc[all_data_df["pp_number"] == pp+1]
    for block_select in blocks: 
    
        data_df = sub_data_df.loc[sub_data_df["block_count"] == block_select] # 60 frames per trial, 150 trials per block: 9000 rows
        
        #Do the classification
        
        """Put the conditions to ordinal values"""
        Cond_cat = data_df[['Affect']]
        data_df_cond_encoded = ordinal_encoder.fit_transform(Cond_cat)
        print(np.unique(ordinal_encoder.categories_))
        data_df.insert(2, "Cond_binary", data_df_cond_encoded, True)
        
        all_mean = []
        all_std = []
        
        for this_frame in np.unique(data_df['frame']):
            """First step: do classification on 1 frame only: here the frame is put to 20"""
            data_f = data_df.loc[data_df["frame"] ==this_frame]
            data_f = data_f.reset_index()
            
            # split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
            # for train_index, test_index in split.split(data_f, data_f["Affect"]):
            #     strat_train_set = data_f.loc[train_index]
            #     strat_test_set = data_f.loc[test_index]
            """Just as a check whether it worked correctly: 
                - train_set: 80 pos, 80 neg
                - test_set: 20 pos, 20 neg"""
            
            #strat_test_set["Affect"].value_counts() / len(strat_test_set)
            #strat_train_set["Affect"].value_counts() / len(strat_train_set)
            
            """create the x and y variables for strat_train_set & strat_test_set"""
            AU_col = [col for col in data_f.columns if ('AU' in col and '_r' in col)] 
            train_x, train_y = data_f[AU_col], data_f['Cond_binary']
            #test_x, test_y = strat_test_set[AU_col], strat_test_set['Cond_binary']
            #transform the y-data to integers, as this is often required by ML algorithms
            train_y = train_y.astype(np.uint8)
            #test_y = test_y.astype(np.uint8)
            
            """Feature-scaling: might have to do this (see book page 109)
            - important: fit the scalers to training set only
            - is this correctly done? """
            scaler.fit(train_x)
            train_x_standardized = pd.DataFrame.from_records(scaler.transform(train_x))
            train_x_standardized.columns = train_x.columns
            # test_x_standardized = pd.DataFrame.from_records(scaler.transform(test_x))
            # test_x_standardized.columns = test_x.columns
            
            
            """The actual classfication"""
            from sklearn import svm
            classifier = svm.SVC(kernel = 'linear', C = 1)
            # classifier.fit(train_x, train_y)
            # print("The score when trained on train-set, tested on test-set")
            # print(classifier.score(test_x, test_y)
            
            
            """work with k-fold cross-validation"""
            print("\nNow doing k-fold cross-validation")
            from sklearn.model_selection import cross_val_score
            cross_scores = cross_val_score(classifier, train_x_standardized, train_y, cv=10, scoring="accuracy")
            
            
            display_scores(cross_scores)
            mean, std = end_scores(cross_scores)
            
            all_mean.append(mean)
            all_std.append(std)
        store_all_means[pp, block_select, :] = all_mean
        store_all_std[pp, block_select, :] = all_std
        #Plot the classification outcome
        plt.errorbar(frames, all_mean, yerr = all_std, fmt = formats[block_select], color = colors[block_select], label = 'block {}'.format(block_select))
        
    axs.plot(frames, np.repeat(0.5, len(frames)), color = 'red', label = 'Chance level')
    axs.plot([15,15],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='appeared')
    axs.plot([46,46],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle('Classification for pp {}'.format(pp))
    # fig.savefig('FperF_allAUs_pp{}.png'.format(pp))
    
#%%
fig, axs = plt.subplots(1, 1)
axs.set_ylim(0, 1.05)
for block in blocks: 
    means = np.mean(store_all_means[:, block, :], axis = 0)
    stds = np.std(store_all_means[:, block, :], axis = 0)
    plt.errorbar(frames, means, yerr = stds, fmt = formats[block], color = colors[block], label = 'block {}'.format(block))
axs.plot(frames, np.repeat(0.5, len(frames)), color = 'red', label = 'Chance level')
axs.plot([15,15],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='appeared')
axs.plot([46,46],[0,5], lw = 2, linestyle ="dashed", color ='y', label ='disappeared')
handles, labels = axs.get_legend_handles_labels()
handles = np.delete(handles, [1, 2])
labels = np.delete(labels, [1, 2])
fig.legend(handles, labels, loc="upper right")
fig.suptitle('Classification scores averaged over all pp')

