# SageMaker

All of the utilities that are used for SageMaker has been added here. This will for a testing ground for
generating many of the Machine learning utilities genertaed as a result of SageMaker. The most important
aspect of this set of code is this is supposed to be run is "script mode". Meaning that we will _not_ be
using SageMaker Studio. This will allow us to properly maintain and version code within code repositories
like git and not be limited by Jupyter Notebook instances within SageMaker Studio.


# Requirements

1. Access to AWS: Remember that SageMaker ispart of the AWS stack. To access it, you will need to have proper
   AWS credentials. You will also need to generate your access credentials and configuration information in
   the right folders (in Mac and *nix systems, this is typically within your `~/.aws`).
2. Create the proper IAM role that will allow you access to `S3` and `ECR` for AWS. Once you have created this
   role you will need to get the `Arn` of this role and save it somehwere. You can get it from the AWS console
   or with the AWS CLI command `aws iam list-roles` and finding the corresponding the ARN corresponding to the
   role that you created. This information is present in the file `config/awsConfig/awsConfig.json`. This file 
   is not tracked by Git. An example if this file is present in the `config/awsConfig/awsConfig-example.json` 
   file, with which you will be able to create the requried file with youe credentials.
3. Docker: AWS makes heavy use of docker. Make sure that docker is installed and operational. Make sure that you
   are able to run a docker container that can run a tensorflow application on your system. For docker to access
   the GPU, you will need to install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) properly. Follow
   the instructions in the nvidia docker page and make sure that you are able to run the docker container with
   proper GPU access.
4. Virtual Environment: Make sure that you are using a virtul environment. An example setup for the virtual 
   environment is shown in the section `Setting up your Virtual Environment`

# Creating an S3 bucket

You can create an S3 bucket with the AWS CLI command:

`aws s3api create-bucket --bucket <bucket name> --region=<region name> --create-bucket-configuration LocationConstraint=<region name>`

For example, I have created one with the command:

`aws s3api create-bucket --bucket sankha-sagemaker-test --region=ap-southeast-1 --create-bucket-configuration LocationConstraint=ap-southeast-1`

Check whether the bucket is available:

`aws s3 ls | grep <part of the bucket name>`

# Setting up your Virtual Environment

Make sure that you have the required packages. For my installation, I have CUDA 11.0, CUDA ToolKit 11.0, 
cuDNN 8.0 installed on an Ubuntu 20.04 focal system. For that, as of the writing, the best wheel packages
for TensorFlow is: Tensroflow 2.4.1. I am currently using Pythn 3.8. Create a virtual environment with the 
folliwing commands:

```bash
python3 -m venv env
source env/bin/activate
pip3 install ──upgrade pip
pip3 install wheel
pip3 install tensorflow=2.4.1
pip3 install numpy scipy matplotlib jupyter pyyaml
pip3 install sagemaker boto3
```

The code has been tested on a GPU with the fllowing specifications:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     │
├───────────────────────────────┬──────────────────────┬──────────────────────┤
│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ Volatile Uncorr. ECC │
│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │
│                               │                      │               MIG M. │
╞═══════════════════════════════╪══════════════════════╪══════════════════════╡
│   0  GeForce RTX 2070    Off  │ 00000000:01:00.0  On │                  N/A │
│  0%   47C    P8    21W / 175W │   1456MiB /  7979MiB │      1%      Default │
│                               │                      │                  N/A │
└───────────────────────────────┴──────────────────────┴──────────────────────┘
```


For some reason, the current version of tensorflow overflows in memory usage and
errors out for RTX 2070 seres. For that reason, you will need to add the following
lines to your TensorFlow code to prevent that from happening.

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
```

# Steps

## 1. Generate Data

The first thing that you want to do is to generate data. Script for generating data is present in the file
[`src/part_1_genertaeData/generateData.py`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_01_genertaeData/generateData.py)
A convinient make option is present to generate the data. Use the command:

`make generateData`

to generate data within the `./data` folder. If this fodler isn't present, it will be created. Also, this data is pushed
to your S3 bucket. The name of the bucket is present in the file `config/awsConfig/awsConfig.json`. Check to make sure
that this file is indeed generated using the command:

`aws s3 ls <bucket-name> --recursive`

## 2. Train Locally

Now, make sure that you are able to train the entire thing locally. Code for this is present in the file
[`src/part_02_runLocal`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_02_runLocal).
A convinient make option is present:

`make runLocal`

This will train the model with the data created in the previous location ...

## 3. Train Locally With Args

Now, make sure that you are able to train the entire thing locally with command lien arguments. Code for 
this is present in the file
[`src/part_03_runLocalArgs`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_03_runLocalArgs).
A convinient make option is present:

`make runLocalArgs`

This will train the model with the data created in the previous location ...

## 4. Train Locally With Sagemaker

Now we shall run the entire script with SageMaker on the local machine. The code for this is present
in the folder
[`src/part_04_runLocalSageMaker`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_04_runLocalSageMaker).
A convinient make option is present:

`make runLocalSageMaker`

This will train the model with the data created in the previous location ...

## 5. Train Locally With SagemakerS3

Now we shall run the entire script with SageMaker on the local machine. This is exactly the same as the previous
problem except that this one actually downloads the data from an S3 bucket provided. This will then be used for
training the model. All the necessary plumbingis automatically done in the background by SageMaker, and hence, you
will not have to do any of this at your end.

The code for this is present
in the folder
[`src/part_05_runLocalSageMakerS3`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_05_runLocalSageMakerS3).
A convinient make option is present:

`make runLocalSageMakerS3`

This will train the model. However, the data will be downloaded from the provided S3 container and will be put into a
particular volume that is generated by SageMaker within the container. Once this data has been generated, SageMaker will
then run the container, and get the results. Finally, it will take the folder in which the model has been created, and
will push the result into another S3 bucket associated with the run. 

## 6. Train on a remote machine with SageMaker

Now the scripy will be deployed to a remote machine instead of a local machine. SageMaker will create a new instance,
create a Docker container, push the docker container to the instance that it just created, download the data that it
requires into the respective folders, and then run your code. Finally, the model file will be compressed, and sent to
a particular S3 bucket associated with the run. Finally, the instance will be torn down.

The code for this is present in the folder
[`src/part_06_runRemoteSageMakerS3`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_06_runRemoteSageMakerS3).
A convinient make option is present:

`make runRemoteSageMaker`

## 7. Batch Inference

Check out the [README.md](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_07_batchInference/README.md) file for this section. This is a little involved and thus
cannot be described in this small section.

## 8. Hyperparameer Optimization Using SageMaker

This will allow you to create a training script as well as train a set of runs using the hyperparameter optimization tool
provided by Sagemaker. Sagemaker will allow you to run a hyperparameter training job and start running those jobs on different
instances directly, and store all runs, along with metadata in S3 buskets, so that they can be used later for your own
experiments for deployments. Example code for doing this is present in the file 
[`src/part_08_hpo/hop.py`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_08_hop/hpo.py)

A convinient Make option is present. Just use the command:

`make hpo`

# References

Number of GPUs per machine for AWS (2021-05-28):

| instance_type   | #GPUs | GPU Memory (GB) |
|-----------------|-------|-----------------|
| ml.p3.2xlarge   |    1  |              16 |
| ml.p3.8xlarge   |    4  |              64 |
| ml.p3.16xlarge  |    8  |             128 |
| ml.p3.24xlarge  |    8  |             256 |

# Authors

Sankha S. Mukherjee - Initial work (2021)

# License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details


# References

1. [(Video) Setting the ARN for your local machine](https://www.youtube.com/watch?v=K3ngZKF31mc)
2. [(Video) Using Script mode in Amazon SageMaker](https://www.youtube.com/watch?v=x94hpOmKtXM)
3. [AWS SageMaker Script Mode Examples GitHub Repo](https://github.com/aws-samples/amazon-sagemaker-script-mode)
4. [AWS SageMaker Local Mode Examples](https://github.com/aws-samples/amazon-sagemaker-local-mode)
4. [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)
5. [AWS Pricing Charts](https://aws.amazon.com/ec2/instance-types/p3/)
