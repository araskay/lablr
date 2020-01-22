import keras.models as km
import numpy as np
import sklearn.model_selection as skms
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
import sklearn.covariance as skc
import flickrutils
import imageutils
import pickle # used to save the data for the web app
import cv2

'''
Validate the model and save train/test data and PCA model
in pickle files to be used by the web app.

Note: this version loads train/test data into memory from
python pickle files. If using generators, use flickr_vae_gen_validate.py
'''

def bgr2rgb(srcBGR):
    return(cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB))
    
datadir='/scratch/mkayvan/flickr/normalized2000ma'
modeldir='/scratch/mkayvan/flickr/normalized2000ma'
image_shape=(128,128,3)

labels = flickrutils.get_labels(datadir+'/labels.txt')
x_train = np.load(datadir+'/x_train.npy')
y_train = np.load(datadir+'/y_train.npy')
x_test = np.load(datadir+'/x_test.npy')
y_test = np.load(datadir+'/y_test.npy')

x_train = np.float32(x_train)/255.0
x_test = np.float32(x_test)/255.0

'''
# test the encoder/decoder
encoder = km.load_model(modeldir+'/vae_encoder.h5')
decoder = km.load_model(modeldir+'/vae_decoder.h5')

sample = x_test[0]
plt.figure(1)
plt.imshow(imageutils.bgr2rgb(sample),interpolation='bilinear')
plt.title('Origianl')
plt.show()

lv_sample = encoder.predict(np.expand_dims(sample,axis=0))

sample_rec = np.squeeze(decoder.predict(lv_sample[2]))

plt.figure(2)
plt.imshow(imageutils.bgr2rgb(sample_rec),interpolation='bilinear')
plt.title('Reconstructed')
plt.show()
'''

# label prediction accuracy
encoder = km.load_model(modeldir+'/vae_encoder.h5')

lv_train = encoder.predict(x_train)[2]

pca = skd.PCA()
pca.fit(lv_train)
#print(pca.explained_variance_ratio_)

lv_train_pca = pca.transform(lv_train)

lv_test = encoder.predict(x_test)[2]
lv_test_pca = pca.transform(lv_test)

nclass=len(labels)

empirical_covs = []

# don't need to use pca for MCD-- can just run on the LVs
for i in range(nclass):
    empirical_covs.append(skc.EmpiricalCovariance().fit(lv_train_pca[y_train==i,]))
    

d_robust = np.zeros((lv_test_pca.shape[0],nclass))
for i in range(nclass):
    d_robust[:,i] = np.array(empirical_covs[i].mahalanobis(lv_test_pca))
    
predicted_class = np.argmin(d_robust,axis=-1)
predicted_labels = np.argsort(d_robust,axis=-1)[:,:5]

print('First label accuracy =',sum(predicted_class == y_test)/len(y_test))

count = 0
for i in range(len(y_test)):
    count += (y_test[i] in predicted_labels[i,:])

print('First five labels accuracy =',count/len(y_test))

# save data to file for the web app


with open(modeldir+'/lv_train.dat','wb') as f:
    pickle.dump(lv_train,f)
with open(modeldir+'/labels.dat','wb') as f:
    pickle.dump(labels,f)
with open(modeldir+'/pca.dat','wb') as f:
    pickle.dump(pca,f)
with open(modeldir+'/y_train.dat','wb') as f:
    pickle.dump(y_train,f)
