# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:26:43 2022

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
from Functions import delete_pp_block, display_scores, end_scores, select_columns

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

#%%
analysis = 'meanAU' # analysis_type should be 'FperF' or 'meanAU'

def select_frames(analysis_type = 'FperF', data = cleaned_data): 
    if analysis_type == 'FperF': 
        frame_selection = np.unique(data.Frame_count).astype(int)
        frameselection_names = frame_selection
        n_subsets = frameselection_names.shape[0]
    elif analysis_type == 'meanAU': 
        frame_selection = [np.arange(0, 15, 1), np.arange(15, 45, 1), np.arange(45, 60, 1)]
        frameselection_names = ['F1-15', 'F16-45', 'F46-60']
        n_subsets = len(frameselection_names)
    return frame_selection, frameselection_names, n_subsets

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
k_folds = 5
n_reps = 1

# frame selection: array when FperF; list when meanAU    
frame_selection, frameselection_names, n_subsets = select_frames(analysis_type = analysis)
participants = np.unique(cleaned_data.pp_number).astype(int)
blocks = np.unique(cleaned_data.block_count).astype(int)
seed_values = np.random.randint(0, 100000, n_reps)

#Create empty arrays to store all the obtained mean accuracies for each pp. in each block at each frame_selection and within each repetition
store_all_means = np.empty([n_reps, participants.shape[0], blocks.shape[0], n_subsets])
store_all_std = np.empty([n_reps, participants.shape[0], blocks.shape[0], n_subsets])

for rep, seed in zip(range(n_reps), seed_values):
    print("we're at repetition {}".format(rep))
    #ipp: starts from 0; pp starts from 1
    for ipp, pp in zip(participants-1, participants): 
        print("we're at pp {}".format(pp))
        #select the data relevant for this participant 
        pp_data = cleaned_data.loc[cleaned_data["pp_number"] == pp]
        #iblock: starts from 0, block: starts from 1
        for iblock, block in zip(blocks, blocks+1): 
            # for pp. 28 and 30 in block 2 data not used 
            if (pp == 28 and block == 2) or (pp == 30 and block == 2): 
                store_all_means[rep, ipp, iblock, :] = np.nan
                store_all_std[rep, ipp, iblock, :] = np.nan
            else: 
                block_data = pp_data[pp_data.block_count == iblock]
                final_data = take_mean(start_data = block_data, analysis_type = analysis)
                
                all_mean = []
                all_std = []
                for selected_frames in frameselection_names: 
                    subset_data = final_data.loc[final_data.Frame_count == selected_frames]
                    subset_data = subset_data.reset_index()
                    
                    x, y = subset_data[AU_cols], subset_data['Cond_binary']
                    #transform the y-data to integers, as this is often required by ML algorithms
                    y = y.astype(np.uint8)
                    
                    #Create the actual classifier (non-trained)
                    classifier = svm.SVC(kernel = 'linear', C = 1)
                    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state = seed)
                    # cv = RepeatedStratifiedKFold(n_splits = k_folds, n_repeats = 2)
                    cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", n_jobs = -1)
                    # cross_scores = cross_val_score(classifier, x, y, cv=cv, scoring="roc_auc", n_jobs = -1)
                    mean, std = end_scores(cross_scores)
                    
                    all_mean.append(mean)
                    all_std.append(std)
                store_all_means[rep, ipp, iblock, :] = all_mean
                store_all_std[rep, ipp, iblock, :] = all_std

#Store the results in a file!
array_dir = os.path.join(os.getcwd(), 'Stored_results')
if not os.path.isdir(array_dir): os.makedirs(array_dir)
np.save(os.path.join(array_dir, "mean_accuracies_{}reps_{}.npy".format(n_reps, analysis)), store_all_means)

#%%Plot the results 
correction = 'fdr' # should be holm or fdr or bonferroni

frames_corrected_for = np.arange(0, 60, 1)
blocks = np.array([0, 1, 2])


formats = ['--o', '--o', '--o']
colors = ['red', 'green', 'orange']
averaged_means = np.nanmean(store_all_means, axis = 0)

if analysis == 'meanAU': 
    fig, axs = plt.subplots(1, n_subsets, sharex = True, sharey = True)
    axs[0].set_ylim(0.3, 1.10)
    p_values = np.empty((3, n_subsets))
    for iblock in blocks: 
        ax = axs[iblock]
        ax.set_title("Block  {}".format(iblock+1))
        ax.axhline(y = 0.5, label = "Chance level", color = 'r')    
        labels = ['F 0-15', 'F 15-45', 'F 45-60']
        for isubset in range(n_subsets): 
            means = np.nanmean(averaged_means[:, iblock, isubset], axis = 0)
            stds = np.nanstd(averaged_means[:, iblock, isubset], axis = 0)
            # ax.errorbar(isubset, means, yerr = stds, fmt = formats[iblock], color = colors[iblock], label = '')
            x = averaged_means[:, iblock, isubset] 
            x = x[~np.isnan(x)]
            ax.boxplot(x, positions = [isubset])
            statistic, p_value = wilcoxon(x - 0.50, alternative = 'greater')
            """Problem: p-values do take nan into account I think!"""
            p_values[iblock, isubset] = p_value
        if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, :], alpha=0.05, method='indep', is_sorted=False)
        elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'holm')
        elif correction == 'bonferroni': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, :], alpha = 0.05, method = 'bonferroni')
        if np.any(signif_frames == True): 
            ax.plot(np.arange(0, 3, 1)[signif_frames], np.repeat(1.05, np.sum(signif_frames)), '*', color = 'black', markersize = 5)

    fig.suptitle("Classification scores averaged over all pp \nCorrection method: {}"
                 .format(correction))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'AverageAU_{}_below85deleted{}_{}fold_{}reps.png'
                             .format(correction, delete_below85, k_folds, n_reps)))

elif analysis == 'FperF': 
    fig, axs = plt.subplots(1, 1)
    axs.set_ylim(0.35, 1.05)
    if blocks.shape[0] != 3:  axs.set_ylim(0.35, 0.6)
    significant_frames = np.array([])
    p_values = np.empty((blocks.shape[0], n_subsets))
    averaged_means = np.nanmean(store_all_means, axis = 0)

    for iblock, block in zip(blocks, blocks+1): 
        means = np.nanmean(averaged_means[:, iblock, :], axis = 0)
        stds = np.nanstd(averaged_means[:, iblock, :], axis = 0)
        plt.errorbar(frame_selection, means, yerr = stds, fmt = formats[iblock], color = colors[iblock], label = 'block {}'.format(block))
        for frame in frame_selection: 
            x = averaged_means[:, iblock, frame] 
            x = x[~np.isnan(x)]
            statistic, p_value = wilcoxon(x - 0.50, alternative = 'greater')
            p_values[iblock, frame] = p_value

        if correction == 'fdr': signif_frames, corrected_pvals = multitest.fdrcorrection(p_values[iblock, frames_corrected_for], alpha=0.05, method='indep', is_sorted=False)
        elif correction == 'holm': signif_frames, corrected_pvals, b, c = multitest.multipletests(p_values[iblock, frames_corrected_for], alpha = 0.05, method = 'holm')
        if np.any(signif_frames == True): axs.plot(frames_corrected_for[signif_frames], np.repeat(0.45-block*0.025, frames_corrected_for[signif_frames].shape[0]), 'o', color = colors[iblock], markersize = 5)
        
           
    axs.plot(frame_selection, np.repeat(0.5, n_subsets), color = 'black', label = 'Chance level')
    axs.plot([15,15],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='appeared')
    axs.plot([46,46],[0,5], lw = 0.5, linestyle ="dashed", color ='black', label ='disappeared')
    handles, labels = axs.get_legend_handles_labels()
    handles = np.delete(handles, [1, 2])
    labels = np.delete(labels, [1, 2])
    fig.legend(handles, labels, loc="center right")
    fig.suptitle('Classification scores averaged over all pp')
    axs.set_title("Correction method: {}, frames included: {}-{}".format(correction, 
                                                                         frames_corrected_for[0]+1, 
                                                                         frames_corrected_for[-1]+1))
    fig.savefig(os.path.join(os.getcwd(), 'Classification_plots', 'F_per_F_{}_frames{}to{}_below85deleted{}.png'.format(correction, 
                                                                              frames_corrected_for[0]+1, 
                                                                              frames_corrected_for[-1]+1, delete_below85)))
















