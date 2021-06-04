import json, yaml
import sagemaker, boto3, os

from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner


if __name__ == "__main__":

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
        output_path           = 's3://sankha-sagemaker-test/models',
        hyperparameters       = {
            'epochs'   : 30,
            'lr'       : 1e-3,
            'decay'    : 1e-6,
            'momentum' : 0.9
        },
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
    print('| Starting a HPO Job')
    print('+-----------------------------------------')

    channels = {
        'training'   : 's3://sankha-sagemaker-test/training',
        'validation' : 's3://sankha-sagemaker-test/validation',
    }

    hpoRange = {
        "lr"       : ContinuousParameter(1e-4, 1e-2),
        "decay"    : ContinuousParameter(1e-9, 1e-6),
        "momentum" : ContinuousParameter(0.5, 0.99),
    }

    objectiveName = "average validation loss"
    objectiveType = "Minimize"
    metricDefinitions = [
        {
            "Name": "average validation loss",
            "Regex": "Validation loss:  ([0-9\\.]+)",
        }
    ]

    tuner = HyperparameterTuner(
        tfEstimator,
        objectiveName,
        hpoRange,
        metricDefinitions,
        max_jobs=4,
        max_parallel_jobs=2,
        objective_type=objectiveType,
    )

    tuner.fit(inputs=channels)



