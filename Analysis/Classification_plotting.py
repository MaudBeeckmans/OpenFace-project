# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:06:39 2022

@author: maudb
"""


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


#%%Plot the results
delete_below85 = True 
analysis = 'FperF'
classification_cross_blocks = True
k_folds = 5
n_reps = 1000
results_path = os.path.join(os.getcwd(), 'Stored_results')
if classification_cross_blocks == True: 
    averaged_means = np.load(os.path.join(results_path, "mean_accuracies_{}_crossblocks.npy".format(analysis)))
    blocks = np.array([0, 1])
    k_folds = 0
    n_reps = 1
else: averaged_means = np.load(os.path.join(results_path, "mean_accuracies_1000reps_{}.npy".format(analysis)))

correction = 'fdr' # should be holm or fdr or bonferroni

frames_corrected_for = np.arange(15, 60, 1)

if analysis == 'meanAU': frame_selection, frameselection_names, n_subsets = select_frames(data = averaged_means, analysis_type = analysis)
else: 
    frame_selection = np.arange(0, 60, 1)
    n_subsets = frame_selection.shape[0]

participants = np.arange(0, averaged_means.shape[0], 1)
blocks = np.arange(0, averaged_means.shape[1], 1)
n_blocks = blocks.shape[0]


formats = ['--o', '--o', '--o']
colors = ['red', 'green', 'orange']

title = 'Decoding accuracy for {} analysis \nCorrection method: Benjamini/Hochberg'.format(analysis)

if analysis == 'meanAU': 
    fig, axs = plt.subplots(1, n_blocks, sharex = True, sharey = True)
    axs[0].set_ylim(0.3, 1.10)
    p_values = np.empty((3, 2))
    statistics = np.empty((3, 2))
    for iblock in blocks: 
        ax = axs[iblock]
        ax.set_title("Block  {}".format(iblock+1), fontsize = 12)
        ax.axhline(y = 0.5, label = "Chance level", color = 'r')    
        labels = ['F 0-15', 'F 15-45', 'F 45-60']
        for isubset in range(1, n_subsets, 1): 
            means = np.nanmean(averaged_means[:, iblock, isubset], axis = 0)
            stds = np.nanstd(averaged_means[:, iblock, isubset], axis = 0)
            # ax.errorbar(isubset, means, yerr = stds, fmt = formats[iblock], color = colors[iblock], label = '')
            z = averaged_means[:, iblock, isubset] 
            z = z[~np.isnan(z)]
            ax.boxplot(z, positions = [isubset], showmeans = True)
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            p_values[iblock, isubset-1] = p_value
            statistics[iblock, isubset-1] = statistic
            if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, :], alpha=0.05, method='indep', is_sorted=False)
            elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'holm')
            elif correction == 'bonferroni': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'bonferroni')
            if np.any(signif_frames == True): 
                ax.plot(np.arange(1, 3, 1)[signif_frames], np.repeat(1.05, np.sum(signif_frames)), '*', color = 'black', markersize = 5)
    fig.suptitle(title, fontsize = 13)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_ylabel('decoding accuracy', fontsize = 12)
    fig.legend(handles, labels, loc="center right", fontsize = 15)
    fig.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'Final', 'AverageAU_{}_below85deleted{}_{}fold_{}reps.png'
                             .format(correction, delete_below85, k_folds, n_reps)))

elif analysis == 'FperF': 
    fig, axs = plt.subplots(1, 1)
    axs.set_ylim(0.35, 1.05)
    if blocks.shape[0] != 3:  axs.set_ylim(0.35, 0.6)
    significant_frames = np.array([])
    p_values = np.empty((blocks.shape[0], n_subsets))
    statistics = np.empty((blocks.shape[0], n_subsets))

    for iblock, block in zip(blocks, blocks+1): 
        means = np.nanmean(averaged_means[:, iblock, :], axis = 0)
        stds = np.nanstd(averaged_means[:, iblock, :], axis = 0)
        plt.errorbar(frame_selection, means, yerr = stds, fmt = formats[iblock], color = colors[iblock], label = 'block {}'.format(block))
        for frame in frame_selection: 
            z = averaged_means[:, iblock, frame] 
            z = z[~np.isnan(z)]
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            statistics[iblock, frame] = statistic
            p_values[iblock, frame] = p_value

        if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, frames_corrected_for], alpha=0.05, method='indep', is_sorted=False)
        elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, frames_corrected_for], alpha = 0.05, method = 'holm')
        if np.any(signif_frames == True): axs.plot(frames_corrected_for[signif_frames], np.repeat(0.45-block*0.025, frames_corrected_for[signif_frames].shape[0]), '*', color = colors[iblock], markersize = 5)
        
           
    axs.plot(frame_selection, np.repeat(0.5, n_subsets), color = 'black', label = 'Chance level')
    axs.plot([14,14],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='appeared')
    axs.plot([45,45],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='disappeared')
    handles, labels = axs.get_legend_handles_labels()
    handles = np.delete(handles, [1, 2])
    labels = np.delete(labels, [1, 2])
    axs.set_ylabel('decoding accuracy', fontsize = 12)
    fig.legend(handles, labels, loc="center right", fontsize = 12)
    fig.suptitle(title, fontsize = 13)
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'Final', 'F_per_F_{}_frames{}to{}_below85deleted{}_{}folds_{}reps.png'.format(correction, 
                                                                              frames_corrected_for[0]+1, 
                                                                              frames_corrected_for[-1]+1, delete_below85, 
                                                                              k_folds, n_reps)))

#%%
#Results FperF
#P-values: 
    # Block 1: 
        # array([0.623, 0.63 , 0.65 , 0.754, 0.76 , 0.807, 0.765, 0.797, 0.695,
        #        0.713, 0.602, 0.643, 0.868, 0.701, 0.56 , 0.539, 0.602, 0.701,
        #        0.663, 0.701, 0.656, 0.553, 0.281, 0.117, 0.257, 0.324, 0.35 ,
        #        0.426, 0.318, 0.391, 0.293, 0.377, 0.275, 0.193, 0.08 , 0.198,
        #        0.246, 0.447, 0.623, 0.293, 0.344, 0.293, 0.246, 0.426, 0.269])
    # block 2: 
        # array([0.017, 0.015, 0.013, 0.03 , 0.034, 0.02 , 0.009, 0.012, 0.009,
        #        0.052, 0.008, 0.009, 0.006, 0.012, 0.012, 0.01 , 0.008, 0.005,
        #        0.002, 0.005, 0.012, 0.036, 0.076, 0.095, 0.082, 0.03 , 0.071,
        #        0.108, 0.056, 0.079, 0.016, 0.005, 0.003, 0.004, 0.007, 0.01 ,
        #        0.004, 0.004, 0.016, 0.141, 0.095, 0.05 , 0.03 , 0.012, 0.002])
    # block 3: 
    #     array([0.068, 0.089, 0.104, 0.17 , 0.148, 0.183, 0.174, 0.032, 0.   ,
    #            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    #            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    #            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    #            0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
    




