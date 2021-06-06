import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow                  import keras
from tensorflow.keras            import Sequential
from tensorflow.python.keras     import backend          as K
from tensorflow.keras.layers     import Conv2D, BatchNormalization
from tensorflow.keras.layers     import Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.losses     import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

import numpy as np
import argparse, os, json, codecs
import tarfile

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

def getPrevModel(folder):

    try:
        # always save the model in the folder '1' within the supplied
        # folder
        model = keras.models.load_model(os.path.join(folder, '1'))

        return model
    except Exception as e:
        print(f'Unable to load the model: {e}')
        return None

def preparePrevModel(folder):

    # we expect that there is only a single file within this folder
    fileNames = [f for f in os.listdir( folder ) if f.endswith('.tar.gz')]
    fileName  = fileNames[0]
    fileName  = os.path.join(folder, fileName)

    if fileName.endswith("tar.gz"):
        tar = tarfile.open(fileName, "r:gz")
        tar.extractall(folder)
        tar.close()
    
    return

def getTrainingData(folder):

    X_train = np.load(os.path.join(folder, 'X_train.npy'))
    y_train = np.load(os.path.join(folder, 'y_train.npy'))

    return X_train, y_train

def getTestingData(folder):

    X_test = np.load(os.path.join(folder, 'X_test.npy'))
    y_test = np.load(os.path.join(folder, 'y_test.npy'))

    return X_test, y_test

def getArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',     type = int,   default = 10                                  )
    parser.add_argument('--lr',         type = float, default = 1e-3                                )
    parser.add_argument('--batch_size', type = int,   default = 128                                 )
    parser.add_argument('--gpu-count',  type = int,   default = os.environ['SM_NUM_GPUS']           )
    parser.add_argument('--model-dir',  type = str,   default = os.environ['SM_MODEL_DIR']          )

    parser.add_argument('--training',   type = str,   default = os.environ['SM_CHANNEL_TRAINING']   )
    parser.add_argument('--validation', type = str,   default = os.environ['SM_CHANNEL_VALIDATION'] )
    parser.add_argument('--prevmodel',  type = str,   default = os.environ['SM_CHANNEL_PREVMODEL']  )

    args, _ = parser.parse_known_args()

    return args


# This is present in this example ...
# https://github.com/aws-samples/amazon-sagemaker-script-mode/blob/master/tf-sentiment-script-mode/sentiment.py
def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
               history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4) 

def main(args):

    epochs     = args.epochs
    lr         = args.lr
    batch_size = args.batch_size
    model_dir  = args.model_dir

    # Input channels
    train_dir     = args.training
    test_dir      = args.validation
    prevModel_dir = args.prevmodel

    print(f'train_dir -----------------------: {train_dir}')
    print(f'Files in the training folder-----: {os.listdir( train_dir )}')

    print(f'test_dir ------------------------: {test_dir}')
    print(f'Files in the testing folder------: {os.listdir( test_dir )}')

    print(f'prevModel_dir -------------------: {prevModel_dir}')
    print(f'Files in the prev model folder---: {os.listdir( prevModel_dir )}')

    preparePrevModel(prevModel_dir)

    print(f'prevModel_dir -------------------: {prevModel_dir}')
    print(f'Files in the prev model folder---: {os.listdir( prevModel_dir )}')


    X_train, y_train = getTrainingData(train_dir)
    X_test, y_test   = getTestingData(test_dir)

    print('-------------[ Getting the old model ]---------------------')
    model     = getPrevModel( prevModel_dir )
    if model is None:
        model     = getModel()
        print('-------------[ Old model not generated ]--------------')
        print('-------------[ New model not generated ]--------------')
    else:
        print('-------------[ Old model successfully generated ]---------------------')

    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss = categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    history = model.fit( X_train, y_train, batch_size=batch_size, 
                    validation_data=(X_test, y_test), validation_batch_size=batch_size, 
                    epochs=epochs, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Validation loss: ', score[0])
    print('Validation accuracy: ', score[1])

    os.makedirs(os.path.join(model_dir, 'scores'))
    with open(os.path.join(model_dir, 'scores', 'scores.json'), 'w') as fOut:
        json.dump( score, fOut )

    os.makedirs(os.path.join(model_dir, 'history'))
    save_history( os.path.join(model_dir, 'history', 'history.p'), history)


    os.makedirs(os.path.join(model_dir, '1'))
    model.save( os.path.join(model_dir, '1'), 'myModel' )

    
    return

if __name__ == "__main__":

    args  = getArgs()
    main(args)