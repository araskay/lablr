'''
Create train/test sets and save in python pickle files as numpy arrays.
'''


import pandas as pd
import numpy as np
#import skimage.io as skiio
import cv2

# the following is the csv file including a list of the images to use
imagelist = '/scratch/mkayvan/flickr/normalized2000ma.csv'

# the following directory is where the pickle files are saved to
imagedir = '/scratch/mkayvan/flickr/normalized2000ma'

imagesize = (128,128,3)

data = pd.read_csv(imagelist,header=None)

n = len(data)
np.random.seed(0) # for reproducibility
indices = np.random.permutation(n)

# use 90% of the data for training and 10% for testing
i_split = int(n*0.9)

# memory allocation
x_train = np.zeros((len(range(0,i_split)),)+imagesize,dtype=int)
y_train = np.zeros(len(range(0,i_split)),dtype=int)

x_test = np.zeros((len(range(i_split,n)),)+imagesize,dtype=int)
y_test = np.zeros(len(range(i_split,n)),dtype=int)

print('Memory allocated successfully.')

for i in range(0,len(y_train)):
    x_train[i,] = cv2.imread(data.iloc[indices[i]][0], cv2.IMREAD_UNCHANGED)
    y_train[i]=data.iloc[indices[i]][1]

for i in range(0,len(y_test)):
    x_test[i,] = cv2.imread(data.iloc[indices[i+i_split]][0], cv2.IMREAD_UNCHANGED)
    y_test[i]=data.iloc[indices[i+i_split]][1]

np.save(imagedir+'/x_train.npy',x_train)
np.save(imagedir+'/x_test.npy',x_test)
np.save(imagedir+'/y_train.npy',y_train)
np.save(imagedir+'/y_test.npy',y_test)
