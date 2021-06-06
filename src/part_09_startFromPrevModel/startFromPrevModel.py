import json, yaml
import sagemaker, boto3, os

from sagemaker.tensorflow import TensorFlow


if __name__ == "__main__":

    role = json.load(open( 'config/awsConfig/awsConfig.json' ))['arn']

    tfEstimator = TensorFlow(
        entry_point           = 'runRemoteSageMakerTrainScript.py',
        source_dir            = 'src/part_09_startFromPrevModel/src',
        role                  = role,
        instance_count        = 1,
        instance_type         = 'ml.p3.2xlarge',
        framework_version     = '2.4.1',
        py_version            = 'py37',
        script_mode           = True,
        hyperparameters       = {'epochs': 2},
        output_path           = 's3://sankha-sagemaker-test/models',
    )

    # Number of GPUs per machine:
    #
    # | instance_type   | #GPUs | GPU Memory (GB) |
    # |-----------------|-------|-----------------|
    # | ml.p3.2xlarge   |    1  |              16 |
    # | ml.p3.8xlarge   |    4  |              64 |
    # | ml.p3.16xlarge  |    8  |             128 |
    # | ml.p3.24xlarge  |    8  |             256 |

    print('+-----------------------------------------')
    print('| Starting a Training Job')
    print('+-----------------------------------------')

    tfEstimator.fit({
        'training'   : 's3://sankha-sagemaker-test/training',
        'validation' : 's3://sankha-sagemaker-test/validation',
        'prevmodel'  : 's3://sankha-sagemaker-test/models/tensorflow-training-210605-0233-004-8d20e36a/output',
    })
