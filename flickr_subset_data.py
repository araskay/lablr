import pandas as pd
import flickrutils

'''
Create a new list of images (saved in a csv file) from an existing
list using a subset of the labels-- useful to train the VAE model
on a selected subset of labels.
'''

# files hardcoded-- boooo!
original_list = '/scratch/mkayvan/flickr/normalized2000ma.csv'
new_list = '/scratch/mkayvan/flickr/normalized2000ma_10labels.csv'

original_labels_file = '/scratch/mkayvan/flickr/labels.txt'
new_labels_file = '/scratch/mkayvan/flickr/normalized2000ma_subset/labels.txt'

original_labels = flickrutils.get_labels(original_labels_file)
new_labels = flickrutils.get_labels(new_labels_file)

data = pd.read_csv(original_list,header=None)

indices = [i for i in range(len(original_labels)) if original_labels[i] in new_labels]


l = [(data.iloc[i,0], indices.index(data.iloc[i,1])) for i in range(len(data)) if data.iloc[i,1] in indices]

new_data = pd.DataFrame(l)

new_data.to_csv(new_list,index=False,header=False)

# or if you feel keen to create your own csv file from scratch, use the following:
'''
f = open(new_list,'w')
for i in range(len(data)):
    if data.iloc[i,1] in indices:
        f.write(data.iloc[i,0]+','+str(indices.index(data.iloc[i,1]))+'\n')

f.close()
'''


