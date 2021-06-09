import os
import numpy as np

import logging

def main():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    inpFolder   = '/opt/ml/processing/rawData'
    trainFolder = "/opt/ml/processing/intermediate/train"
    testFolder  = "/opt/ml/processing/intermediate/test"

    os.makedirs(trainFolder, exist_ok=True)
    os.makedirs(testFolder, exist_ok=True)

    logger.info('Reading the data from the input files ...')

    test_images  = np.load(os.path.join(inpFolder, 'test_images.npy'))
    test_labels  = np.load(os.path.join(inpFolder, 'test_labels.npy'))
    train_images = np.load(os.path.join(inpFolder, 'train_images.npy'))
    train_labels = np.load(os.path.join(inpFolder, 'train_labels.npy'))

    # Do the transformations
    X_train = (train_images/255).astype('float32')
    X_test = (test_images/255).astype('float32')

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # Create the OHE for the data labels
    y_train = np.zeros((len(train_labels), 10)).astype('float32')
    y_train[  np.arange(len(train_labels)),  train_labels  ] = 1

    y_test = np.zeros((len(test_labels), 10)).astype('float32')
    y_test[  np.arange(len(test_labels)),  test_labels  ] = 1

    np.save( os.path.join(testFolder,  'X_test.npy') , X_test)
    np.save( os.path.join(testFolder,  'y_test.npy') , y_test)

    np.save( os.path.join(trainFolder, 'X_train.npy'), X_train)
    np.save( os.path.join(trainFolder, 'y_train.npy'), y_train)

    logger.info('Files written to the output paths ...')

    return

if __name__ == "__main__":
    main()
