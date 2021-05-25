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
   role that you created. We will create an environment variable `AWS_ARN` that will allow us to securely use
   this information within our code, without worring about whether we are saving this information within GitHub.
   This information is also present in the file `config/awsConfig/awsConfig.json`. This file is not tracked by 
   Git. An example if this file is present in the `config/awsConfig/awsConfig-example.json` file, with which
   you will be able to create the requried file with youe credentials.
3. Docker: AWS makes heavy use of docker. Make sure that docker is installed and operational. Make sure that you
   are able to run a docker container that can run a tensorflow application on your system. For docker to access
   the GPU, you will need to install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) properly. Follow
   the instructions in the nvidia docker page and make sure that you are able to run the docker container with
   proper GPU access.
4. Virtual Environment: Make sure that you are using a virtul environment. An example setup for the virtual 
   environment is shown in the section `Setting up your Virtual Environment`


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
pip3 install numpy scipy matplotlib jupyter
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
[`src/part_1_genertaeData/generateData.py`](https://github.com/sankhaMukherjee/sageMaker/blob/master/src/part_1_genertaeData/generateData.py)
A convinient make option is present to generate the data. Use the command:

`make generateData`

to generate data within the `./data` folder. If this fodler isn't present, it will be created.

# Authors

Sankha S. Mukherjee - Initial work (2021)

# License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details


# References

1. [Setting the ARN for your local machine](https://www.youtube.com/watch?v=K3ngZKF31mc)
2. [Using Script mode in Amazon SageMaker](https://www.youtube.com/watch?v=x94hpOmKtXM)