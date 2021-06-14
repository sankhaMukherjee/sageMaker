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
            "training"   : trainData,
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

    description = tfEstimator.jobs[-1].describe()

    print('------------[description of the training job]-----------')
    print(description)

    description = {
        'TrainingJobName': 'tensorflow-training-2021-06-14-02-21-30-008', 
        'TrainingJobArn': 'arn:aws:sagemaker:ap-southeast-1:387826921024:training-job/tensorflow-training-2021-06-14-02-21-30-008', 
        'ModelArtifacts': {
            'S3ModelArtifacts': 's3://sankha-sagemaker-test/models/tensorflow-training-2021-06-14-02-21-30-008/output/model.tar.gz'}, 
            'TrainingJobStatus': 'Completed', 
            'SecondaryStatus': 'Completed', 
            'HyperParameters': {
                'epochs': '2', 
                'model_dir': '"s3://sankha-sagemaker-test/models/tensorflow-training-2021-06-14-02-21-30-008/model"', 
                'sagemaker_container_log_level': '20', 
                'sagemaker_job_name': '"tensorflow-training-2021-06-14-02-21-30-008"', 
                'sagemaker_program': '"runRemoteSageMakerTrainScript.py"', 
                'sagemaker_region': '"ap-southeast-1"', 
                'sagemaker_submit_directory': '"s3://sankha-sagemaker-test/tensorflow-training-2021-06-14-02-21-30-008/source/sourcedir.tar.gz"'
            }, 
            'AlgorithmSpecification': {
                'TrainingImage': '763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-training:2.4.1-gpu-py37', 
                'TrainingInputMode': 'File', 
                'EnableSageMakerMetricsTimeSeries': True
            }, 
            'RoleArn': 'arn:aws:iam::387826921024:role/sankha-sagemaker-test-1', 
            'InputDataConfig': [
                {
                    'ChannelName': 'training', 
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix', 
                            'S3Uri': 's3://sankha-sagemaker-test/training', 
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }, 
                    'CompressionType': 'None', 
                    'RecordWrapperType': 'None'
                }, 
                {
                    'ChannelName': 'validation', 
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix', 
                            'S3Uri': 's3://sankha-sagemaker-test/validation', 
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }, 
                    'CompressionType': 'None', 
                    'RecordWrapperType': 'None'
                }
            ], 
            'OutputDataConfig': {
                'KmsKeyId': '', 
                'S3OutputPath': 's3://sankha-sagemaker-test/models'
            }, 
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge', 
                'InstanceCount': 1, 
                'VolumeSizeInGB': 30
            }, 
            'StoppingCondition': {'MaxRuntimeInSeconds': 86400}, 
            'CreationTime': datetime.datetime(2021, 6, 14, 10, 21, 30, 633000, tzinfo=tzlocal()), 
            'TrainingStartTime': datetime.datetime(2021, 6, 14, 10, 23, 50, 796000, tzinfo=tzlocal()), 
            'TrainingEndTime': datetime.datetime(2021, 6, 14, 10, 27, 3, 971000, tzinfo=tzlocal()), 
            'LastModifiedTime': datetime.datetime(2021, 6, 14, 10, 27, 16, 871000, tzinfo=tzlocal()), 
            'SecondaryStatusTransitions': [
                {
                    'Status': 'Starting', 'StartTime': datetime.datetime(2021, 6, 14, 10, 21, 30, 633000, tzinfo=tzlocal()), 
                    'EndTime': datetime.datetime(2021, 6, 14, 10, 23, 50, 796000, tzinfo=tzlocal()), 
                    'StatusMessage': 'Preparing the instances for training'
                }, 
                {
                    'Status': 'Downloading', 'StartTime': datetime.datetime(2021, 6, 14, 10, 23, 50, 796000, tzinfo=tzlocal()), 
                    'EndTime': datetime.datetime(2021, 6, 14, 10, 24, 2, 921000, tzinfo=tzlocal()), 
                    'StatusMessage': 'Downloading input data'
                }, 
                {
                    'Status': 'Training', 
                    'StartTime': datetime.datetime(2021, 6, 14, 10, 24, 2, 921000, tzinfo=tzlocal()), 
                    'EndTime': datetime.datetime(2021, 6, 14, 10, 26, 55, 431000, tzinfo=tzlocal()), 
                    'StatusMessage': 'Training image download completed. Training in progress.'
                }, 
                {
                    'Status': 'Uploading', 
                    'StartTime': datetime.datetime(2021, 6, 14, 10, 26, 55, 431000, tzinfo=tzlocal()), 
                    'EndTime': datetime.datetime(2021, 6, 14, 10, 27, 3, 971000, tzinfo=tzlocal()), 
                    'StatusMessage': 'Uploading generated training model'
                }, 
                {
                    'Status': 'Completed', 
                    'StartTime': datetime.datetime(2021, 6, 14, 10, 27, 3, 971000, tzinfo=tzlocal()), 
                    'EndTime': datetime.datetime(2021, 6, 14, 10, 27, 3, 971000, tzinfo=tzlocal()), 
                    'StatusMessage': 'Training job completed'
                }
            ], 
            'EnableNetworkIsolation': False, 
            'EnableInterContainerTrafficEncryption': False, 
            'EnableManagedSpotTraining': False, 
            'TrainingTimeInSeconds': 193, 
            'BillableTimeInSeconds': 193, 
            'DebugHookConfig': {
                'S3OutputPath': 's3://sankha-sagemaker-test/models', 
                'CollectionConfigurations': []
            }, 
            'ProfilerConfig': {
                'S3OutputPath': 's3://sankha-sagemaker-test/models', 
                'ProfilingIntervalInMilliseconds': 500
            }, 
            'ProfilerRuleConfigurations': [
                {
                    'RuleConfigurationName': 'ProfilerReport-1623637290', 
                    'RuleEvaluatorImage': '972752614525.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-debugger-rules:latest', 
                    'VolumeSizeInGB': 0, 
                    'RuleParameters': {
                        'rule_to_invoke': 'ProfilerReport'
                    }
                }
            ], 
            'ProfilerRuleEvaluationStatuses': [
                {
                    'RuleConfigurationName': 'ProfilerReport-1623637290', 
                    'RuleEvaluationJobArn': 'arn:aws:sagemaker:ap-southeast-1:387826921024:processing-job/tensorflow-training-2021-0-profilerreport-1623637290-2473c70b', 
                    'RuleEvaluationStatus': 'NoIssuesFound', 
                    'LastModifiedTime': datetime.datetime(2021, 6, 14, 10, 27, 16, 858000, tzinfo=tzlocal())
                }
            ], 
            'ProfilingStatus': 'Enabled', 
            'ResponseMetadata': {
                'RequestId': 'a784f53d-8b6f-4d9f-8304-eb92b6883065', 
                'HTTPStatusCode': 200, 
                'HTTPHeaders': {
                    'x-amzn-requestid': 'a784f53d-8b6f-4d9f-8304-eb92b6883065', 
                    'content-type': 'application/x-amz-json-1.1', 
                    'content-length': '3866', 
                    'date': 'Mon, 14 Jun 2021 02:27:16 GMT'
                }, 
            'RetryAttempts': 0
        }
    }

