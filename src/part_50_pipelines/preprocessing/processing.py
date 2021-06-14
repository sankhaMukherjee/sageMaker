from sagemaker.workflow            import steps
from sagemaker.processing          import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing  import SKLearnProcessor
import os, json

def getProcessor():

    config = json.load(open('config/awsConfig/awsConfig.json'))
    bucket = config['s3bucket']
    role   = config['arn']

    sklearnProcessor = SKLearnProcessor(
        framework_version = "0.20.0",
        role              = role,
        instance_type     = "ml.m5.xlarge",
        instance_count    = 1,
    )

    return sklearnProcessor

def getProcessingStep(experimentName):

    config = json.load(open('config/awsConfig/awsConfig.json'))
    bucket = config['s3bucket']

    preProcessingStep = steps.ProcessingStep(
        name      = experimentName,
        processor = getProcessor(),
        code="src/part_50_pipelines/preprocessing/processData.py",
        inputs=[
            ProcessingInput(
                source      = f"s3://{bucket}/data/pipelines/rawData", 
                destination = "/opt/ml/processing/rawData"
            ),
        ],
        outputs=[
            ProcessingOutput(
                source      = "/opt/ml/processing/intermediate/test",
                output_name = f"test", 
            ),

            ProcessingOutput(
                source      = "/opt/ml/processing/intermediate/train",
                output_name = f"train", 
            ),
        ],
    )

    return preProcessingStep

def main():

    skearnProcessor = getProcessor()

    config = json.load(open('config/awsConfig/awsConfig.json'))
    bucket = config['s3bucket']
    role   = config['arn']


    skearnProcessor.run(
        code="src/part_50_pipelines/preprocessing/processData.py",
        inputs=[
            ProcessingInput(
                source      = f"s3://{bucket}/data/pipelines/rawData", 
                destination = "/opt/ml/processing/rawData"
            ),
        ],
        outputs=[
            ProcessingOutput(
                source      = "/opt/ml/processing/intermediate/test",
                output_name = f"test", 
            ),

            ProcessingOutput(
                source      = "/opt/ml/processing/intermediate/train",
                output_name = f"train", 
            ),
        ],
        
    )

    description = skearnProcessor.jobs[-1].describe()

    print('---------[description of the processign job]----------')
    print(description)

    return

if __name__ == "__main__":
    main() 

    result = {

        'ProcessingInputs': [
            {
                'InputName': 'input-1', 
                'AppManaged': False, 
                'S3Input': {
                    'S3Uri': 's3://sankha-sagemaker-test/data/pipelines/rawData', 
                    'LocalPath': '/opt/ml/processing/rawData', 
                    'S3DataType': 'S3Prefix', 
                    'S3InputMode': 'File', 
                    'S3DataDistributionType': 'FullyReplicated', 
                    'S3CompressionType': 'None'
                }
            }, 
            {
                'InputName': 'code', 
                'AppManaged': False, 
                'S3Input': {
                    'S3Uri': 's3://sagemaker-ap-southeast-1-387826921024/sagemaker-scikit-learn-2021-06-09-04-28-59-399/input/code/processData.py', 
                    'LocalPath': '/opt/ml/processing/input/code', 
                    'S3DataType': 'S3Prefix', 
                    'S3InputMode': 'File', 
                    'S3DataDistributionType': 'FullyReplicated', 
                    'S3CompressionType': 'None'
                }
            }
        ], 

        'ProcessingOutputConfig': {
            'Outputs': [
                {
                    'OutputName': 'test', 
                    'S3Output': {
                        'S3Uri': 's3://sagemaker-ap-southeast-1-387826921024/sagemaker-scikit-learn-2021-06-09-04-28-59-399/output/test', 
                        'LocalPath': '/opt/ml/processing/intermediate/test', 
                        'S3UploadMode': 'EndOfJob'
                    }, 
                    'AppManaged': False
                }, 
                {
                    'OutputName': 'train', 
                    'S3Output': {
                        'S3Uri': 's3://sagemaker-ap-southeast-1-387826921024/sagemaker-scikit-learn-2021-06-09-04-28-59-399/output/train', 
                        'LocalPath': '/opt/ml/processing/intermediate/train', 
                        'S3UploadMode': 'EndOfJob'
                    }, 'AppManaged': False
                }
            ]
        }, 
        
        'ProcessingJobName': 'sagemaker-scikit-learn-2021-06-09-04-28-59-399', 
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1, 
                'InstanceType': 'ml.m5.xlarge', 
                'VolumeSizeInGB': 30}
            }, 
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400
            }, 
            'AppSpecification': {
                'ImageUri': '121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3', 
                'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/processData.py']
            }, 
            'RoleArn': 'arn:aws:iam::387826921024:role/sankha-sagemaker-test-1', 
            'ProcessingJobArn': 'arn:aws:sagemaker:ap-southeast-1:387826921024:processing-job/sagemaker-scikit-learn-2021-06-09-04-28-59-399', 
            'ProcessingJobStatus': 'Completed', 
            'ProcessingEndTime': datetime.datetime(2021, 6, 9, 12, 33, 7, 692000, tzinfo=tzlocal()), 
            'ProcessingStartTime': datetime.datetime(2021, 6, 9, 12, 32, 49, 205000, tzinfo=tzlocal()), 
            'LastModifiedTime': datetime.datetime(2021, 6, 9, 12, 33, 7, 702000, tzinfo=tzlocal()), 
            'CreationTime': datetime.datetime(2021, 6, 9, 12, 29, 0, 130000, tzinfo=tzlocal()), 
            'ResponseMetadata': {
                'RequestId': 'adef1d60-26f6-4756-b8b3-05e4667d7265', 
                'HTTPStatusCode': 200, 
                'HTTPHeaders': {
                    'x-amzn-requestid': 'adef1d60-26f6-4756-b8b3-05e4667d7265', 
                    'content-type': 'application/x-amz-json-1.1', 
                    'content-length': '1963', 'date': 'Wed, 09 Jun 2021 04:33:42 GMT'
                }, 
                'RetryAttempts': 0
            }
        }


