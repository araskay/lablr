# -*- coding: utf-8 -*-

import pickle, cv2
import numpy as np
import keras.models as km
import matplotlib.pyplot as plt
import os
import sklearn.covariance as skc
import imageutils, flickrutils
import time

maindir='.'

with open(maindir+'/lv_train.dat','rb') as f:
    lv_train=pickle.load(f)
with open(maindir+'/labels.dat','rb') as f:
    labels=pickle.load(f)
with open(maindir+'/pca.dat','rb') as f:
    pca=pickle.load(f)
with open(maindir+'/y_train.dat','rb') as f:
    y_train=pickle.load(f)

encoder = km.load_model(maindir+'/vae_encoder_gen_maxpool_16fm.h5')

image_shape=(128,128,3)

lv_train_pca = pca.transform(lv_train)

nclass=len(labels)

empirical_covs = []
for i in range(nclass):
    empirical_covs.append(skc.EmpiricalCovariance().fit(lv_train_pca[y_train==i,]))

def get_lables(image,uploadfolder='.'):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    normalized = imageutils.normalize_image(img, image_shape)
    x = np.expand_dims(normalized,axis=0)
    x = np.float32(x)/255.0
    
    lv_x = encoder.predict(x)[2]
    lv_x_pca = pca.transform(lv_x)

    d_robust = np.zeros(nclass)
    for i in range(nclass):
        d_robust[i] = empirical_covs[i].mahalanobis(lv_x_pca)
        
    #predicted_class = np.argmin(d_robust,axis=-1)
    predicted_labels = np.argsort(d_robust,axis=-1)
    
    # select top 5 labels
    toplabels = [labels[i] for i in predicted_labels[:5]]
    
    # plots
    uploadfolder = os.path.abspath(uploadfolder)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    plotfilename_latent='latent_space'+timestr+'.png'
    plotfile_latent = uploadfolder+'/'+plotfilename_latent

    plotfilename_pca='pca'+timestr+'.png'
    plotfile_pca = uploadfolder+'/'+plotfilename_pca

    # (a) plot class centroids and dispersions in the PCA domain
    n = 20 # number of most related labels to plot
    mean_class = np.zeros((n,lv_train_pca.shape[1]))
    dispersion_class = np.zeros(n)
    
    for i in range(n):
        mean_class[i,:] = np.mean(lv_train_pca[y_train==predicted_labels[i],],axis=0)
        #dispersion_class[i]=np.mean(empirical_covs[predicted_labels[i]].mahalanobis(lv_train_pca[y_train==predicted_labels[i],]))
        dispersion_class[i]= np.linalg.det(empirical_covs[predicted_labels[i]].covariance_)
    
    dispersion_class = flickrutils.normalize_to01(dispersion_class)
    plt.figure()
    for i in range(n):
        plt.scatter(mean_class[i,0],mean_class[i,1], label=str(labels[predicted_labels[i]]), s=(20+dispersion_class[i]*20)**2, alpha=0.5)
        plt.annotate(str(labels[predicted_labels[i]]),(mean_class[i,0],mean_class[i,1]))
    plt.scatter(lv_x_pca[0,0],lv_x_pca[0,1], c='red',marker='x',alpha=1)
    plt.annotate('Input Image',(lv_x_pca[0,0],lv_x_pca[0,1]))
    
    plt.xlabel('PC1 ('+str(int(pca.explained_variance_ratio_[0]*100))+'% variance explained)')
    plt.ylabel('PC2 ('+str(int(pca.explained_variance_ratio_[1]*100))+'% variance explained)')
    plt.savefig(plotfile_latent)
    #plt.show()


    # (b) plot pca of the predicted labels
    n = 3 # number of most related labels to plot
    plt.figure()
    for i in predicted_labels[:n]:
        plt.scatter(lv_train_pca[y_train==i,0],lv_train_pca[y_train==i,1], label=str(labels[i]), alpha=0.3)

    plt.scatter(lv_x_pca[0,0],lv_x_pca[0,1], c='red',marker='x',alpha=1)
    plt.annotate('Input Image',(lv_x_pca[0,0],lv_x_pca[0,1]))

    plt.legend()
    plt.xlabel('PC1 ('+str(int(pca.explained_variance_ratio_[0]*100))+'% variance explained)')
    plt.ylabel('PC2 ('+str(int(pca.explained_variance_ratio_[1]*100))+'% variance explained)')
    plt.savefig(plotfile_pca)
    #plt.show()

    return toplabels,plotfilename_latent,plotfilename_pca
