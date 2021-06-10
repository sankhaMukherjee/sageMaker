import json
import sagemaker, boto3, os

from sagemaker.tensorflow     import TensorFlow
from sagemaker.workflow.steps import TrainingStep


def getEstimator():

    role = json.load(open( 'config/awsConfig/awsConfig.json' ))['arn']

    tfEstimator = TensorFlow(
        entry_point           = 'runRemoteSageMakerTrainScript.py',
        source_dir            = 'src/part_06_runRemoteSageMakerS3/src',
        role                  = role,
        instance_count        = 1,
        instance_type         = 'ml.p3.2xlarge',
        framework_version     = '2.4.1',
        py_version            = 'py37',
        script_mode           = True,
        hyperparameters       = {'epochs': 2},
        output_path           = 's3://sankha-sagemaker-test/models',
    )

    return tfEstimator

def getTrainingStep(experimentName, trainData, testData):

    trainingStep = TrainingStep(
        name      = experimentName,
        estimator = getEstimator(),
        inputs    = {
            "train"      : trainData,
            "validation" : testData
        },
    )

    return trainingStep

if __name__ == "__main__":

    tfEstimator = getEstimator()

    print('+-----------------------------------------')
    print('| Starting a Training Job')
    print('+-----------------------------------------')

    tfEstimator.fit({
        'training'   : 's3://sankha-sagemaker-test/training',
        'validation' : 's3://sankha-sagemaker-test/validation',
    })
