import keras.models as km
import numpy as np
import sklearn.model_selection as skms
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
import sklearn.covariance as skc
import flickrutils
import imageutils
import pickle # used to save the data for the web app
import pandas as pd
import vae_gen_lib


'''
Validate the model and save train/test data and PCA model
in pickle files to be used by the web app.

Note: this version loads train/test data images from disk using a generator.
If loading the entire train/test data from pickle files into memory,
use flickr_vae_validate.py (recommended only on HPC clusters or you have
a lot of memory).
'''

datadir='/scratch/mkayvan/flickr/normalized2000ma'
modeldir='/scratch/mkayvan/flickr/normalized2000ma_subset'
image_shape=(128,128,3)

labels = flickrutils.get_labels(datadir+'/labels.txt')

data_train = pd.read_csv(datadir+'/data_train.csv',header=None)
data_test = pd.read_csv(datadir+'/data_test.csv',header=None)


# calculate label prediction accuracy
encoder = km.load_model(modeldir+'/vae_encoder_gen_maxpool_16fm.h5')


batch_size = 32

# Important: shuffle must be false for prediction
training_gen = vae_gen_lib.DataGenerator(data_train,
                             batch_size=batch_size,
                             dim=image_shape,
                             shuffle=False)

test_gen = vae_gen_lib.DataGenerator(data_test,
                         batch_size=batch_size,
                         dim=image_shape,
                         shuffle=False)

lv_train = encoder.predict_generator(training_gen,
                                     use_multiprocessing=True,
                                     workers=8)[2]

pca = skd.PCA()
pca.fit(lv_train)
#print(pca.explained_variance_ratio_)

lv_train_pca = pca.transform(lv_train)

lv_test = encoder.predict_generator(test_gen,
                                    use_multiprocessing=True,
                                    workers=8)[2]

lv_test_pca = pca.transform(lv_test)

nclass=len(labels)

empirical_covs = []

# create numpy arrays of labels for ease of use
'''due to use of generators, the number of predictions (e.g. lv_train.shape[0])
may not match the actual number of train/test samples.
This needs to be accounted for.'''
y_train = np.array(data_train.iloc[:lv_train.shape[0],1])
y_test = np.array(data_test.iloc[:lv_test.shape[0],1])

# don't need to use pca for MCD-- can just run on the LVs
for i in range(nclass):
    empirical_covs.append(skc.EmpiricalCovariance().fit(lv_train_pca[y_train==i,]))
    

d_robust = np.zeros((lv_test_pca.shape[0],nclass))
for i in range(nclass):
    d_robust[:,i] = np.array(empirical_covs[i].mahalanobis(lv_test_pca))

predicted_class = np.argmin(d_robust,axis=-1)
predicted_labels = np.argsort(d_robust,axis=-1)[:,:10]

print('First label accuracy =',sum(predicted_class == y_test)/len(y_test))

count = 0
for i in range(len(y_test)):
    count += (y_test[i] in predicted_labels[i,:])

print('First ten labels accuracy =',count/len(y_test))

# save data to file for the web app


with open(modeldir+'/lv_train.dat','wb') as f:
    pickle.dump(lv_train,f)
with open(modeldir+'/labels.dat','wb') as f:
    pickle.dump(labels,f)
with open(modeldir+'/pca.dat','wb') as f:
    pickle.dump(pca,f)
with open(modeldir+'/y_train.dat','wb') as f:
    pickle.dump(y_train,f)
