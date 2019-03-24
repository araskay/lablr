import keras.models as km
import numpy as np
import sklearn.model_selection as skms
import sklearn.decomposition as skd
import matplotlib.pyplot as plt
import sklearn.covariance as skc
import flickrutils
import imageutils
import pickle # used to save the data for the web app

'''
Calculate accuracy for a PCA dimensionality reduction model
instead of VAE model. For comparison only.
'''

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

# label prediction accuracy
print('Doing PCA on x_train')
x = np.reshape(x_train,(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
pca_model = skd.PCA(n_components=100)
pca_model.fit(x)
lv_train = pca_model.transform(x)

pca = skd.PCA()
pca.fit(lv_train)
#print(pca.explained_variance_ratio_)

lv_train_pca = pca.transform(lv_train)

lv_test = pca_model.transform(np.reshape(x_test,(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
lv_test_pca = pca.transform(lv_test)

nclass=len(labels)

empirical_covs = []

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

