import keras.utils as ku
import numpy as np
import cv2


class DataGenerator(ku.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim=(128,128,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data # pandas dataframe
        # with first column (data[0]) containing imagefiles
        # and second column (data[1]) containing labels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_temp = self.data.iloc[indexes]

        # Generate data
        X, y = self.__data_generation(data_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim), dtype = float)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(len(data_temp)):
            img = cv2.imread(data_temp.iloc[i][0], cv2.IMREAD_UNCHANGED)
            # Store sample
            X[i,] = np.float32(img)/255.0

            # Store label - do not need for VAE
            #y[i] = data_temp.iloc[i][1]

        return X, X

