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

data_path = os.path.join(os.getcwd(), 'data_processed_concat10to11_AUstatic.rds')
data = pyreadr.read_r(data_path) # also works for RData
data_df = data[None]

#%%Plot AU4 & AU12 over time for visualization (mean values & per trial)

"""Plot the AU4 & AU12 over time for visualization"""
data_smile = data_df.loc[data_df["Condition"] =="smile"]
data_frown = data_df.loc[data_df["Condition"] =="frown"]

AU_list = ['AU04_r', 'AU12_r']
data_cond = [data_frown, data_smile]
for cond in data_cond: 
    for i in range(len(AU_list)): 
        data_mean = []
        for frame in np.unique(cond["Frame_count"]): 
            data_select = cond.loc[cond['Frame_count'] == frame]
            mean_frame = np.mean(data_select[AU_list[i]])
            data_mean.append(mean_frame)
        if i == 0: 
            mean_all = data_mean
        else: 
            mean_all = np.column_stack([mean_all, data_mean])
    if np.all(cond['Condition'] =='frown') == True: 
        frown_mean_all = mean_all
        print('Frown done')
    else: 
        smile_mean_all = mean_all
        print('Smile done')

"""Figure with the mean values for each frame, per condition over time for AU4 & AU12 
    - averaged over trials"""
fig, axs = plt.subplots(1, 2)
Data_list = [smile_mean_all, frown_mean_all]
color = ['r', 'b']
label = ['smile', 'frown']
for count, data in enumerate(Data_list): 
    frames = np.arange(data.shape[0])
    print(frames)
    print(data.shape)
    axs[0].plot(frames, data[:,0], color = color[count], label = label[count])
    axs[1].plot(frames, data[:, 1], color = color[count], label = label[count])
axs[0].set_title("AU_04")
axs[1].set_title("AU_12")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")

"""Figure with the mean values for each frame, per condition over time for AU4 & AU12 
    - for each trial separately"""
fig, axs = plt.subplots(1, 2)
Data_list = [data_smile, data_frown]
color = ['r', 'b']
label = ['smile', 'frown']
for count, data in enumerate(Data_list): 
    print(data.shape)
    axs[0].plot(data["Frame_count"], data['AU04_r'], color = color[count], label = label[count])
    axs[1].plot(data["Frame_count"], data['AU12_r'], color = color[count], label = label[count])
axs[0].set_title("AU_04")
axs[1].set_title("AU_12")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")


#%%Do the classification

"""Put the conditions to ordinal values"""
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
Cond_cat = data_df[['Condition']]
data_df_cond_encoded = ordinal_encoder.fit_transform(Cond_cat)
print(np.unique(ordinal_encoder.categories_))
data_df.insert(2, "Cond_binary", data_df_cond_encoded, True)

"""one hot encoding: to use when working with more than 2 categories
    - doesn't work accurately yet: only encodes everything as 1 now"""
# from sklearn.preprocessing import OneHotEncoder
# cat_encoder = OneHotEncoder()
# cond_1hot = cat_encoder.fit_transform(data_df['Condition'].to_numpy().reshape(1, -1))
# cond_1hot = cond_1hot.toarray()

"""Create some functions to use in the loop"""
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
        
def end_scores(scores): 
    mean = scores.mean()
    std = scores.std()
    return mean, std 

all_mean = []
all_std = []

for this_frame in np.unique(data_df['frame']):
    """First step: do classification on 1 frame only: here the frame is put to 20"""
    data_f = data_df.loc[data_df["frame"] ==this_frame]
    data_f = data_f.reset_index()
    
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(data_f, data_f["Condition"]):
        strat_train_set = data_f.loc[train_index]
        strat_test_set = data_f.loc[test_index]
    """Just as a check whether it worked correctly: 
        - train_set: 80 smile, 80 frown
        - test_set: 20 smile, 20 frown"""
    
    #strat_test_set["Condition"].value_counts() / len(strat_test_set)
    #strat_train_set["Condition"].value_counts() / len(strat_train_set)
    
    """create the x and y variables for strat_train_set & strat_test_set"""
    AU_col = [col for col in data_f.columns if ('AU' in col and '_r' in col)] 
    train_x, train_y = strat_train_set[AU_col], strat_train_set['Cond_binary']
    test_x, test_y = strat_test_set[AU_col], strat_test_set['Cond_binary']
    #transform the y-data to integers, as this is often required by ML algorithms
    train_y = train_y.astype(np.uint8)
    test_y = test_y.astype(np.uint8)
    
    """Feature-scaling: might have to do this (see book page 109)
    - important: fit the scalers to training set only
    - is this correctly done? """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x_standardized = pd.DataFrame.from_records(scaler.transform(train_x))
    train_x_standardized.columns = train_x.columns
    test_x_standardized = pd.DataFrame.from_records(scaler.transform(test_x))
    test_x_standardized.columns = test_x.columns
    
    
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

fig, axs = plt.subplots(1, 1)
axs.set_ylim(0, 1.05)
plt.errorbar(frames, all_mean, yerr = all_std, marker = 'x', color = 'black')
axs.plot(frames, np.repeat(0.5, len(frames)), color = 'red', label = 'Chance level')
handles, labels = axs.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")

    
"""Questions: 
    - Will we compare different models?
      --> the k-fold cross-validation: good to see which models perform best?"""
