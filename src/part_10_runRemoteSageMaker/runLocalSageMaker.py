import json, yaml
import sagemaker, boto3, os

from sagemaker.tensorflow import TensorFlow


configText = '''
[default]
ap-southeast-1
'''

if __name__ == "__main__":

    sess = sagemaker.Session( boto3.session.Session(region_name='ap-southeast-1') )
    role = "arn:aws:iam::387826921024:role/sankha-sagemaker-test-1" #json.load(open( 'config/awsConfig/awsConfig.json' ))['arn']

    training   = 'file://data'
    validation = 'file://data'
    output     = 'file://model'

    tfEstimator = TensorFlow(
                #   entry_point = 'src/part_04_runLocalSageMaker/runLocalSageMaker.py',
                  entry_point = 'runLocalSageMaker.py',
                         role = role,
            sagemaker_session = sess,
               instance_count = 1,
                # instance_type = 'local_gpu',
                # instance_type = 'ml.p3.8xlarge',
                instance_type = 'ml.p3.2xlarge',
            framework_version = '2.4.1',
                   py_version = 'py37',
                  script_mode = True,
              hyperparameters = {'epochs': 1},
            #   output_path = output, environment={"NEW_THING":"xxxxxxxxxxxxxxxxxxxxxx"}

    )

    print('+-----------------------------------------')
    print('| This is something')
    print('+-----------------------------------------')

    # tfEstimator.fit({
    #   'training'   : training,
    #   'validation' : validation,
    # })
    
    tfEstimator.fit({
      'training'   : 's3://sankha-sagemaker-test/training',
      'validation' : 's3://sankha-sagemaker-test/validation',
    })
