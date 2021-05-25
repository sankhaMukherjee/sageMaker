import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras            import Sequential
from tensorflow.keras.layers     import Conv2D, BatchNormalization
from tensorflow.keras.layers     import Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.losses     import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

import numpy as np
import os 

def getModel():

    model = Sequential([

        Conv2D(64, (3,3), padding='same', input_shape=(28,28,1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D((2,2), strides=2),

        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D((2,2), strides=2),
        
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),

        Dense(10, activation='softmax')
    ])

    return model

def getTrainingData(folder):

    X_train = np.load(os.path.join(folder, 'X_train.npy'))
    y_train = np.load(os.path.join(folder, 'y_train.npy'))

    return X_train, y_train

def getTestingData(folder):

    X_test = np.load(os.path.join(folder, 'X_test.npy'))
    y_test = np.load(os.path.join(folder, 'y_test.npy'))

    return X_test, y_test

def main():

    epochs     = 10
    lr         = 1e-3
    batch_size = 128
    model_dir  = 'model'
    train_dir  = 'data'
    test_dir   = 'data'

    X_train, y_train = getTrainingData(train_dir)
    X_test, y_test   = getTestingData(test_dir)

    model     = getModel()
    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss = categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    model.fit( X_train, y_train, batch_size=batch_size, 
                    validation_data=(X_test, y_test), validation_batch_size=batch_size, 
                    epochs=epochs, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Validation loss: ', score[0])
    print('Validation accuracy: ', score[1])

    model.save( model_dir, 'myModel' )



    return

if __name__ == "__main__":
    main()