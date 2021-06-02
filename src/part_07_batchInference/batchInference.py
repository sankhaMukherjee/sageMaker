from sagemaker.tensorflow.model import TensorFlowModel
import json 

def main():

    config = json.load(open('config/awsConfig/awsConfig.json'))
    
    model = TensorFlowModel(
        model_data        = "s3://sankha-sagemaker-test/inference/models/model.tar.gz",
        role              = config['arn'],
        framework_version = '2.4.1'
    )


    transformer = model.transformer(
        instance_count            = 1,
        instance_type             = "ml.p3.2xlarge",
        max_concurrent_transforms = 1,
        max_payload               = 1,
        output_path               = "s3://sankha-sagemaker-test/miniServing/predictions",
        # strategy                  = 'SingleRecord',
    )

    transformer.transform(
        data         = "s3://sankha-sagemaker-test/miniServingJson/X",
        content_type = "application/json",
        logs         = True,
        job_name     = "tensorflow-BI",
        split_type   = None,
    )

    return


if __name__ == "__main__":
    main()

