# -*- coding: utf-8 -*-

'''
Download images from flickr with specifit labels.
'''

import flickrapi
import urllib
import fileutils
import flickrutils

# number of images to download within each label
n = 5000

# images are downloaded to the following directory-- hardcoded booooo!
maindir = '/home/mkayvanrad/scratch/flickr'

# text file including all labels
tags = flickrulits.get_lables(maindir+'/labels.txt')

# Flickr api access key 
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

'''
tags = ['sunset', 'beach', 'water', 'sky', 'flower', 'nature', 'blue', 'night',\
        'white', 'tree', 'green', 'flowers', 'portrait', 'art', 'light',\
        'snow', 'dog', 'sun', 'clouds', 'cat', 'park', 'winter', 'landscape',\
        'street', 'summer', 'sea', 'city', 'trees', 'yellow', 'lake',\
        'christmas', 'people', 'bridge', 'family', 'bird', 'river', 'pink',\
        'house', 'car', 'food', 'macro', 'music', 'moon', 'orange', 'garden']

#tags = ['night','snow','light','moon','music','city','beach','sea','landscape','sunset']
'''

for keyword in tags:
    outputdir=maindir+'/'+keyword
    
    fileutils.createdir(outputdir)
    
    error=True
    while error:
        try:
            photos = flickr.walk(text=keyword,
                                 tag_mode='all',
                                 tags=keyword,
                                 extras='url_c', # if want tags, add 'tags' in a list, i.e., ['url_c','tags']
                                 per_page=100,           # may be you can try different numbers..
                                 sort='relevance')


            urls = []
            for i, photo in enumerate(photos):

                url = photo.get('url_c') # if want tags, tags = photo.get('tags')
                urls.append(url)

                # get only this many urls
                if i > n:
                    break


            count = 0

            for url in urls:
                if not url is None:
                    count += 1
                    urllib.request.urlretrieve(url, outputdir+'/'+'image'+str(count)+'.jpg')

            print(count,'images downloaded for tag',keyword)
            
            error = False
        except:
            print('Something went wrong downloading tag',keyword,'Trying again')
