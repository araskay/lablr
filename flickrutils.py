def get_labels(filename):
    '''
        read lables from text file and return as a list
    '''
    f = open(filename)
    labels = []
    l = f.readline()
    while l:
        labels.append(l.strip())
        l = f.readline()
    return(labels)

def normalize_to01(x):
    '''
        normalize a numpy array to [0,1]
    '''
    minimum = np.min(x)
    maximum = np.max(x)
    return((x-minimum)/(maximum-minimum))