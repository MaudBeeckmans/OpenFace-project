# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:18:15 2022

@author: maudb
"""
import pyreadr, os, random
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm
from statsmodels.stats import multitest
from researchpy import ttest

#%%Statistical_output
analysis = 'FperF'
classification_cross_blocks = True
delete_below85 = True 
results_path = os.path.join(os.getcwd(), 'Stored_results')
if classification_cross_blocks == True: 
    k_folds = 0
    n_reps = 1
    averaged_means = np.load(os.path.join(results_path, "mean_accuracies_{}_crossblocks.npy".format(analysis)))
    
else: 
    k_folds = 5
    n_reps = 1000
    averaged_means = np.load(os.path.join(results_path, "mean_accuracies_1000reps_{}.npy".format(analysis)))


blocks = np.arange(0, averaged_means.shape[1], 1)
n_blocks = blocks.shape[0]
n_subsets = averaged_means.shape[2]


    
if analysis == 'meanAU': 
    p_values = np.empty((n_blocks, n_subsets-1))
    corrected_pvalues = np.empty((n_blocks, n_subsets-1))
    ranks = np.empty((n_blocks, n_subsets-1))
    medians = np.empty((n_blocks, n_subsets-1))
    for iblock in blocks: 
        for subset, isubset in zip(range(1, n_subsets), range(n_subsets-1)):
            z = averaged_means[:, iblock, subset] 
            z = z[~np.isnan(z)]
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            p_values[iblock, isubset] = p_value
            ranks[iblock, isubset] = statistic
            medians[iblock, isubset] = np.median(z)
        signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, :], alpha=0.05, method='indep', is_sorted=False)
        corrected_pvalues[iblock, :] = corrected_pvals
        Z_scores = -norm.ppf(p_values)
        print("\n\nResults block {}".format(iblock+1))
        print("Median values: \n{}".format(np.round(medians[iblock, :], 3)))
        print("Corrected p-values: \n{}".format(np.round(corrected_pvalues[iblock, :], 3)))
        print("ranks (W): \n{}".format(np.round(ranks[iblock, :], 3)))
        print("Z_scores: {}".format(Z_scores))
elif analysis == 'FperF': 
    frames_corrected_for = np.arange(15, 60, 1)
    p_values = np.empty((n_blocks, n_subsets))
    corrected_pvalues = np.empty((n_blocks, frames_corrected_for.shape[0]))
    ranks = np.empty((n_blocks, n_subsets))
    medians = np.empty((n_blocks, n_subsets))
    
    for iblock in blocks: 
        for isubset in range(n_subsets): 
            z = averaged_means[:, iblock, isubset] 
            z = z[~np.isnan(z)]
            statistic, p_value = wilcoxon(z - 0.50, alternative = 'greater')
            p_values[iblock, isubset] = p_value
            ranks[iblock, isubset] = statistic
            medians[iblock, isubset] = np.median(z)
        signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, frames_corrected_for], alpha=0.05, 
                                                                      method='indep', is_sorted=False)            
        corrected_pvalues[iblock, :] = corrected_pvals
        print("\n\nResults block {}".format(iblock+1))
        print("Median values: \n{}".format(np.round(medians[iblock, :], 3)))
        print("Corrected p-values: \n{}".format(np.round(corrected_pvalues[iblock, :], 3)))
        print("ranks (W): \n{}".format(np.round(ranks[iblock, :], 3)))
        reduced_medians = medians[:, frames_corrected_for]
        reduced_ranks = ranks[:, frames_corrected_for]
        reduced_p = p_values[:, frames_corrected_for]
        print("\nInfo on the significant frames")
        print(np.round(reduced_medians[iblock, signif_frames], 3))
        print(np.round(corrected_pvalues[iblock, signif_frames], 3))
        print(np.round(reduced_ranks[iblock, signif_frames], 3))
        print(reduced_p[iblock, signif_frames])
        print(np.round(-norm.ppf(reduced_p[iblock, signif_frames])))
        print('number of significant frames: {}'.format(np.sum(signif_frames)))
        


