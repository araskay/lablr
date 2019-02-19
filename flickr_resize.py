# -*- coding: utf-8 -*-

'''
Resize downloaded images to a common size, and create a csv file
with the path/file_name of all images and their label.
'''

import cv2
import numpy as np
import glob
import imageutils
import fileutils
import flickrutils
#import skimage.io as skiio
#import skimage.transform as skit


maindir='/scratch/mkayvan/flickr'
batchname = 'normalized1500ma'

# number of images to use in each class
n = 1500

target_size = (128, 128, 3)

normalizeddir=maindir+'/'+batchname
fileutils.createdir(normalizeddir)

classes = flickrutils.get_labels(maindir+'/labels.txt')

'''
classes = ['sunset', 'beach', 'water', 'sky', 'flower', 'nature', 'blue', 'night',\
        'white', 'tree', 'green', 'flowers', 'portrait', 'art', 'light',\
        'snow', 'dog', 'sun', 'clouds', 'cat', 'park', 'winter', 'landscape',\
        'street', 'summer', 'sea', 'city', 'trees', 'yellow', 'lake',\
        'christmas', 'people', 'bridge', 'family', 'bird', 'river', 'pink',\
        'house', 'car', 'food', 'macro', 'music', 'moon', 'orange', 'garden']

#classes = ['night','snow','light','moon','music','city','beach','sea','landscape','sunset']
'''


f = open(maindir+'/'+batchname+'.csv','w')

classcount=0
totalcount = 0
for i in range(len(classes)):
    imagedir = maindir+'/'+classes[i]
    images = glob.glob(imagedir+'/*.jpg')
    print(str(len(images)),'images to read in class',classes[i])
    
    classcount = 0
    for imagefile in images:
        
        # only read n image in each class
        if classcount >= n:
            break
        
        #img = skiio.imread(imagefile)
        img = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)

        # if grayscale, skip the image
        if len(img.shape)<3:
            print(imagefile,'is greyscale. Skipping!')
            continue

        #normalized = skit.resize(img,target_size)
        normalized = imageutils.normalize_image(img, target_size)

        if normalized.shape == target_size:
            classcount += 1
            totalcount += 1
            #skiio.imsave(normalizeddir+'/'+'image'+str(totalcount)+'.jpg',normalized)
            cv2.imwrite(normalizeddir+'/'+'image'+str(totalcount)+'.jpg',normalized)
            f.write(normalizeddir+'/'+'image'+str(totalcount)+'.jpg'+','+str(i)+'\n')
            f.flush()
        else:
            print('Normalization failed for',imagefile,'-- Skipping!')
            print('Reached',normalized.shape)

f.close()
print('Total of',totalcount,'images normalized.')


# also save labels into normalizeddir
f = open(normalizeddir+'/labels.txt','w')
for l in classes:
    f.write(l+'\n')
f.close()

