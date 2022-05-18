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

single_row = False

classification_cross_blocks = False
run_number = 10
k_folds = 5
n_reps = 100
results_path = os.path.join(os.getcwd(), 'Stored_results')
if classification_cross_blocks == True: 
    averaged_means = np.load(os.path.join(results_path, "mean_accuracies_{}_crossblocks_run{}.npy".format(analysis, run_number)))
    blocks = np.array([0, 1])
    k_folds = 0
    n_reps = 1
    position = 'lower center'
else: 
    averaged_means = np.load(os.path.join(results_path, "mean_accuracies_{}reps_{}.npy".format(n_reps, analysis)))
    position = 'center right'

correction = 'fdr' # should be holm or fdr or bonferroni

frames_corrected_for = np.arange(15, 60, 1)

if analysis == 'meanAU': 
    frame_selection, frameselection_names, n_subsets = select_frames(data = averaged_means, analysis_type = analysis)
    analysis_text = 'temporal averaging'
else: 
    frame_selection = np.arange(0, 60, 1)
    n_subsets = frame_selection.shape[0]
    analysis_text = 'frame-by-frame'

participants = np.arange(0, averaged_means.shape[0], 1)
blocks = np.arange(0, averaged_means.shape[1], 1)
n_blocks = blocks.shape[0]


formats = ['--o', '--o', '--o']
colors = ['red', 'green', 'orange']

title = 'Classification accuracy for {} analysis'.format(analysis_text)

if analysis == 'meanAU': 
    fig, axs = plt.subplots(1, n_blocks, sharex = True, sharey = True)
    # axs[0].set_ylim(0.3, 1.10)
    p_values = np.empty((3, 2))
    statistics = np.empty((3, 2))
    for iblock in blocks: 
        ax = axs[iblock]
        ax.set_title("Block  {}".format(iblock+1), fontsize = 12)
        ax.axhline(y = 0.5, label = "Chance level", color = 'blue', lw = 0.5)    
        labels = ['F 0-15', 'F 15-45', 'F 45-60']
        for isubset in range(1, n_subsets, 1): 
            means = np.nanmean(averaged_means[:, iblock, isubset], axis = 0)
            stds = np.nanstd(averaged_means[:, iblock, isubset], axis = 0)
            # ax.errorbar(isubset, means, yerr = stds, fmt = formats[iblock], color = colors[iblock], label = '')
            z = averaged_means[:, iblock, isubset] 
            z = z[~np.isnan(z)]
            ax.boxplot(z, positions = [isubset], showmeans = False)
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            # print(np.round(np.mean(z), 3))
            print(np.round(p_value, 3))
            
            p_values[iblock, isubset-1] = p_value
            statistics[iblock, isubset-1] = statistic
        if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, :], alpha=0.05, method='indep', is_sorted=False)
        elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'holm')
        elif correction == 'bonferroni': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'bonferroni')
        
        print(np.round(corrected_pvals, 3))
        print('\n')
        if np.any(signif_frames == True): 
            ax.plot(np.arange(1, 3, 1)[signif_frames], np.repeat(0.35, np.sum(signif_frames)), '*', color = 'black', markersize = 5)
    fig.suptitle(title, fontsize = 13)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_ylabel('decoding accuracy', fontsize = 12)
    fig.legend(handles, labels, loc=position, fontsize = 10)
    fig.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'Final', 'AverageAU_{}_below85deleted{}_{}fold_{}reps_run{}.png'
                              .format(correction, delete_below85, k_folds, n_reps, run_number)))

elif analysis == 'FperF': 
    
    if single_row == True: fig, axs = plt.subplots(1, n_blocks, sharex = True, sharey = False)
    else: fig, axs = plt.subplots(n_blocks, 1, sharex = True, sharey = False)
    
    # axs[0].set_xlim(frames_corrected_for[0]-1, frames_corrected_for[-1]+1)
    y_values = [(0.40, 0.65), (0.40, 0.65), (0.35, 1.05)]
    [axs[iblock].set_ylim(y_value) for iblock, y_value in zip(range(n_blocks), y_values[:n_blocks])]
    significant_frames = np.array([])
    p_values = np.empty((blocks.shape[0], n_subsets))
    corrected_pvalues = np.empty((blocks.shape[0], n_subsets))
    statistics = np.empty((blocks.shape[0], n_subsets))

    for iblock, block in zip(blocks, blocks+1): 
        means = np.nanmean(averaged_means[:, iblock, :], axis = 0)
        stds = np.nanstd(averaged_means[:, iblock, :], axis = 0)
        axs[iblock].errorbar(frame_selection, means, yerr = stds, fmt = formats[iblock], color = colors[iblock])
        for frame in frame_selection: 
            z = averaged_means[:, iblock, frame] 
            z = z[~np.isnan(z)]
            # if frame > 14: axs[iblock].boxplot(z, positions = [frame], showmeans = False, showfliers=False)
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            statistics[iblock, frame] = statistic
            p_values[iblock, frame] = p_value

        if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, frames_corrected_for], alpha=0.05, method='indep', is_sorted=False)
        elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, frames_corrected_for], alpha = 0.05, method = 'holm')
        print("\n {}".format(np.round(corrected_pvals, 3)))
        print(signif_frames)
        if np.any(signif_frames == True): axs[iblock].plot(frames_corrected_for[signif_frames], np.repeat(0.42, frames_corrected_for[signif_frames].shape[0]), '*', color = colors[iblock], markersize = 7)
        
        if single_row == True: axs[iblock].set_title('block{}'.format(block), fontsize = 15)    
        else: axs[iblock].set_ylabel('block{}'.format(block), fontsize = 15)    
           
    [axs[iblock].plot(frame_selection, np.repeat(0.5, n_subsets), lw = 1, color = 'blue', label = 'Chance level') for iblock in range(n_blocks)]
    [axs[iblock].plot([14,14],[0,5], lw = 1, linestyle ="dashed", color ='grey', label ='stimulus onset')for iblock in range(n_blocks)]
    [axs[iblock].plot([45,45],[0,5], lw = 1, linestyle ="dashed", color ='grey', label ='stimulus offset')for iblock in range(n_blocks)]
    handles, labels = axs[1].get_legend_handles_labels()
    
    axs[-1].set_xlabel('frame', fontsize = 15)
    fig.legend(handles, labels, loc=position, fontsize = 15)
    fig.set_size_inches(16, 7)
    # fig.tight_layout()
    
    fig.suptitle(title, fontsize = 15)
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'Final', 'F_per_F_{}_frames{}to{}_below85deleted{}_{}folds_{}reps_singlerow{}_run{}.png'.format(correction, 
                                                                              frames_corrected_for[0]+1, 
                                                                              frames_corrected_for[-1]+1, delete_below85, 
                                                                              k_folds, n_reps, single_row, run_number)))





