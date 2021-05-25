import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras          import utils

import os
import numpy as np

def main():

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X_train = (train_images/255).astype('float32')
    X_test = (test_images/255).astype('float32')

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # Convert to float32 numpy arrays
    numClasses = 10
    y_train = utils.to_categorical(train_labels, numClasses)
    y_test  = utils.to_categorical(test_labels, numClasses)

    print('\nGenerated array shapes:')
    print(f'X_train : {X_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'y_test  : {y_test.shape}')

    if not os.path.exists('data'):
        print('The ./data folder does not exist. Generating the ./data folder')
        os.makedirs('data')

    np.save( 'data/X_train.npy', X_train )    
    np.save( 'data/y_train.npy', y_train )    
    np.save( 'data/X_test.npy', X_test )    
    np.save( 'data/y_test.npy', y_test )

    print('\nGenerated data within the ./data folder:')
    print(os.listdir('data'))
    
    return

if __name__ == "__main__":
    main()
