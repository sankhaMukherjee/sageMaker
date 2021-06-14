from sagemaker                  import image_uris
from sagemaker.processing       import ProcessingInput, ProcessingOutput
from sagemaker.processing       import ScriptProcessor
from sagemaker.tensorflow.model import TensorFlowModel
import json, os
import numpy as np 
import matplotlib.pyplot as plt

config    = json.load(open('config/awsConfig/awsConfig.json'))

def getModel(modelData):

    model = TensorFlowModel(
        model_data         = modelData, 
        role               = config['arn'],
        framework_version  = '2.4.1'
    )

    return model

def getData():

    testX = np.load('data/X_test.npy')
    testy = np.load('data/y_test.npy')

    labels = {
        0 : 'T-shirt/top',
        1 : 'Trouser',
        2 : 'Pullover',
        3 : 'Dress',
        4 : 'Coat',
        5 : 'Sandal',
        6 : 'Shirt',
        7 : 'Sneaker',
        8 : 'Bag',
        9 : 'Ankle boot'
    }

    return testX, testy, labels

def main():

    modelData = 's3://sankha-sagemaker-test/models/tensorflow-training-210605-0233-001-0529300b/output/model.tar.gz'
    model     = getModel( modelData )

    testX, testy, labels = getData()

    return

if __name__ == "__main__":
    main()

