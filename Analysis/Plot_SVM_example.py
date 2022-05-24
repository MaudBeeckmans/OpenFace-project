# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:48:31 2022

@author: maudb
"""

from matplotlib import pyplot as plt 
import numpy as np

weights_cat = np.random.randint(10, 20, 10)
ears_cat = np.random.randint(1, 8, 10)
weights_dog = np.random.randint(15, 35, 10)
ears_dog = np.random.randint(6, 15, 10)



fig, axes = plt.subplots(nrows = 1, ncols = 1)
axes.scatter(weights_dog, ears_dog, color = 'red', marker = 'o', label = 'dogs')
axes.scatter(weights_cat, ears_cat, color = 'blue', marker = 'o', label = 'cats')
#%%
test_point = [22, 8]
axes.scatter(test_point[0], test_point[1], color = 'green', marker = 'o', label = 'test data')
axes.set_ylabel("length ears (cm)")
axes.set_xlabel("weight (kg)")

handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, fontsize = 10)