from sagemaker.inputs              import TrainingInput
# from sagemaker.workflow            import steps
from sagemaker.workflow.pipeline   import Pipeline
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

# from src.part_50_pipelines.preprocessing import processing
from preprocessing import processing
from training      import training

from datetime      import datetime as dt

import json
from time import sleep


def getParameters(inputUri, batchUri):

    params = {}
    params["ProcessingInstanceCount"] = ParameterInteger( name = "ProcessingInstanceCount", default_value = 1)
    params["ProcessingInstanceType"]  = ParameterString(  name = "ProcessingInstanceType",  default_value = "ml.p3.2xlarge")
    params["TrainingInstanceType"]    = ParameterString(  name = "TrainingInstanceType",    default_value = "ml.p3.2xlarge")
    params["ModelApprovalStatus"]     = ParameterString(  name = "ModelApprovalStatus",     default_value = "PendingManualApproval")
    params["InputData"]               = ParameterString(  name = "InputData",               default_value = inputUri)
    params["BatchData"]               = ParameterString(  name = "BatchData",               default_value = batchUri)

    return params

def main():

    config = json.load(open('config/awsConfig/awsConfig.json'))
    bucket = config['s3bucket']
    role   = config['arn']

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    
    # params = getParameters( inputUri, batchUri )

    
    experimentName = f'fashionMNIST-{now}'

    # ----------- [Generate a preprocessing step] -------------------------
    preProcessingStep = processing.getProcessingStep(f'{experimentName}-preprocess')

    # ----------- [Generate a training step] -------------------------
    trainData = TrainingInput( s3_data = preProcessingStep.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri)
    testData  = TrainingInput( s3_data = preProcessingStep.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri)

    trainingStep = training.getTrainingStep(f'{experimentName}-training', trainData, testData)

    # ----------- [Generate the pipeline] -------------------------
    pipeline = Pipeline(
        name = experimentName,
        parameters = [],
        steps = [
            preProcessingStep,
            trainingStep
        ]
    )

    definition    = json.loads(pipeline.definition())
    definitionStr = json.dumps( definition, indent=4, sort_keys=True )
    print( definitionStr )

    print('-------------Sending the pipeline to SageMaker --------------------')
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()


    # ----------------------------------------------------------------------------
    # This section only polls the execution ...
    # ----------------------------------------------------------------------------
    while True:

        # Poll the execution
        
        stepsCompleted = []
        for s in execution.list_steps():
            stepsCompleted.append( f"{s['StepName']:50s} -> {s['StepStatus']}")

        stepsCompleted = '\n'.join( stepsCompleted )
        stepsCompleted = f'Steps Completed: \n {stepsCompleted} \n'

        executionData  = execution.describe()
        name           = executionData['PipelineExecutionDisplayName']
        status         = executionData['PipelineExecutionStatus']

        now           = dt.now().strftime('%Y/%m/%d %H:%M:%S')
        statusReport  = f'{name} : [{status}] at {now}\n{stepsCompleted}'
        print(statusReport)

        if status != 'Executing':
            break

        sleep(2)


    # Confirm that the execution is completed
    execution.wait()
    
    return 

if __name__ == "__main__":
    main()

    # {
    #     'PipelineArn'                  : 'arn:aws:sagemaker:ap-southeast-1:387826921024:pipeline/fashionmnist-2021-06-11--00-17-44', 
    #     'PipelineExecutionArn'         : 'arn:aws:sagemaker:ap-southeast-1:387826921024:pipeline/fashionmnist-2021-06-11--00-17-44/execution/9c937oa6drra', 
    #     'PipelineExecutionDisplayName' : 'execution-1623341866106', 
    #     'PipelineExecutionStatus'      : 'Executing', 
    #     'CreationTime'                 : datetime.datetime(2021, 6, 11, 0, 17, 46, 13000, tzinfo=tzlocal()), 
    #     'LastModifiedTime'             : datetime.datetime(2021, 6, 11, 0, 17, 46, 13000, tzinfo=tzlocal()), 
    #     'CreatedBy'                    : {}, 
    #     'LastModifiedBy'               : {}, 
    #     'ResponseMetadata'             : {
    #         'RequestId'      : 'e236557e-7d75-4e89-b4bb-2bc51a882ffa', 
    #         'HTTPStatusCode' : 200, 
    #         'HTTPHeaders'    : {
    #             'x-amzn-requestid': 'e236557e-7d75-4e89-b4bb-2bc51a882ffa', 
    #             'content-type': 'application/x-amz-json-1.1', 
    #             'content-length': '441', 
    #             'date': 'Thu, 10 Jun 2021 16:17:45 GMT'
    #         }, 
    #         'RetryAttempts': 0
    #     }
    # }

    # {
    #     'StepName'  : 'fashionMNIST-2021-06-11--01-07-42-preprocess', 
    #     'StartTime' : datetime.datetime(2021, 6, 11, 1, 7, 45, 102000, tzinfo=tzlocal()), 
    #     'StepStatus': 'Executing', 
    #     'Metadata': {
    #         'ProcessingJob': {
    #             'Arn': 'arn:aws:sagemaker:ap-southeast-1:387826921024:processing-job/pipelines-7qjz5e7zwzvd-fashionmnist-2021-06-93egbiyqvl'
    #         }
    #     }
    # }



