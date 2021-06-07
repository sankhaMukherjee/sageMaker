from sagemaker.tensorflow.model import TensorFlowModel
import json, os
import numpy as np 
import matplotlib.pyplot as plt

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
    config    = json.load(open('config/awsConfig/awsConfig.json'))
    
    # Create a TensorFlow Model object
    # ---------------------------------
    model = TensorFlowModel(
        model_data         = modelData, 
        role               = config['arn'],
        framework_version  = '2.4.1'
    )

    # Create a predictor 
    # ----------------------
    predictor = model.deploy(
        initial_instance_count = 1, 
        instance_type          = 'ml.c5.xlarge',

    )

    # Get the data
    # -------------------------------------------------
    # testX.shape = (10000, 28, 28, 1)
    # testy.shape = (10000, 10)
    # labels = dictionary, key = int, value = str
    # -------------------------------------------------
    testX, testy, labels = getData()

    inputVal1 = { 'instances': testX[:10].tolist() }
    inputVal2 = testX[:10]

    result1 = predictor.predict( inputVal1 )
    result2 = predictor.predict( inputVal2 )

    print('--- [Generating raw predictions] ----------------------')
    print(result1)
    print(result2)

    for i, (y, yHat1, yHat2) in enumerate(zip(testy, result1['predictions'], result2['predictions'] )):
        y, yHat1, yHat2 = np.argmax(y), np.argmax(yHat1), np.argmax(yHat2)
        print(f'| {i:5d} | {y:3d} | {yHat1:3d} | {yHat2:3d}')
    
    predictor.delete_endpoint()

    
    return 

if __name__ == "__main__":
    main()

