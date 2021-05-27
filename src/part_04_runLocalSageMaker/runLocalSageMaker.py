import json, yaml
import sagemaker, boto3, os

from sagemaker.tensorflow import TensorFlow


configText = '''
[default]
ap-southeast-1
'''

if __name__ == "__main__":

    role = json.load(open( 'config/awsConfig/awsConfig.json' ))['arn']

    training   = 'file://src/part_04_runLocalSageMaker/data'
    validation = 'file://src/part_04_runLocalSageMaker/data'
    output     = 'file://src/part_04_runLocalSageMaker/model'

    tfEstimator = TensorFlow(
        entry_point           = 'runLocalSageMakerTrainScript.py',
        source_dir            = 'src/part_04_runLocalSageMaker/src',
        role                  = role,
        instance_count        = 1,
        instance_type         = 'local_gpu',
        framework_version     = '2.4.1',
        py_version            = 'py37',
        script_mode           = True,
        hyperparameters       = {'epochs': 1},
    )

    print('+-----------------------------------------')
    print('| This is something')
    print('+-----------------------------------------')

    tfEstimator.fit({
        'training'   : training,
        'validation' : validation,
    })
    