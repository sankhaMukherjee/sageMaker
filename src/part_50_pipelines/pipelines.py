from sagemaker.workflow.parameters import ParameterInteger, ParameterString

import json


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

    inputUri = f's3://{bucket}/training'
    batchUri = f's3://{bucket}/validation'

    params = getParameters( inputUri, batchUri )

    print(params)

    return 

if __name__ == "__main__":
    main()


