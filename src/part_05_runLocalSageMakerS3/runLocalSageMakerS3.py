import json, yaml
import sagemaker, boto3, os

from sagemaker.tensorflow import TensorFlow


if __name__ == "__main__":

    role = json.load(open( 'config/awsConfig/awsConfig.json' ))['arn']

    tfEstimator = TensorFlow(
        entry_point           = 'runLocalSageMakerTrainScript.py',
        source_dir            = 'src/part_05_runLocalSageMakerS3/src',
        role                  = role,
        instance_count        = 1,
        instance_type         = 'local_gpu',
        framework_version     = '2.4.1',
        py_version            = 'py37',
        script_mode           = True,
        hyperparameters       = {'epochs': 1},
        output_path           = 's3://sankha-sagemaker-test/runs/models',
    )

    print('+-----------------------------------------')
    print('| Starting a Training Job')
    print('+-----------------------------------------')

    tfEstimator.fit({
        'training'   : 's3://sankha-sagemaker-test/training',
        'validation' : 's3://sankha-sagemaker-test/validation',
    })
