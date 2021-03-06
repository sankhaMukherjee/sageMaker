from sagemaker.tensorflow.model import TensorFlowModel
import json 
from datetime import datetime as dt 

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    config   = json.load(open('config/awsConfig/awsConfig.json'))
    s3bucket = config["s3bucket"]
    
    model = TensorFlowModel(
        model_data        = f"s3://{s3bucket}/inference/models/model.tar.gz",
        role              = config['arn'],
        framework_version = '2.4.1',
        entry_point       = 'src/part_07_batchInference/code/inference.py'
    )


    transformer = model.transformer(
        instance_count            = 1,
        instance_type             = "ml.p3.2xlarge",
        max_concurrent_transforms = 1,
        max_payload               = 1,
        output_path               = f"s3://{s3bucket}/miniServing/predictions",
        # strategy                  = 'SingleRecord',
    )

    transformer.transform(
        data         = f"s3://{s3bucket}/miniServingJson/X",
        content_type = "application/json",
        logs         = True,
        job_name     = f"tensorflow-BI-sankha-{now}",
        split_type   = None,
    )

    return


if __name__ == "__main__":
    main()

