import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras          import utils

import os
import numpy as np

def main():

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X_train = (train_images/255).astype('float32')
    X_test = (test_images/255).astype('float32')

    # Convert to float32 numpy arrays
    numClasses = 10
    y_train = utils.to_categorical(train_labels, numClasses)
    y_test  = utils.to_categorical(test_labels, numClasses)

    if not os.path.exists('data'):
        print('The ./data folder does not exist. Generating the ./data folder')
        os.makedirs('data')

    np.save( 'data/X_train.npy', X_train )    
    np.save( 'data/y_train.npy', y_train )    
    np.save( 'data/X_test.npy', X_test )    
    np.save( 'data/y_test.npy', y_test )

    print('Generated data within the ./data folder:')
    print(os.listdir('data'))
    
    return

if __name__ == "__main__":
    main()
