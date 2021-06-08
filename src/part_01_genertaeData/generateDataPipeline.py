from tensorflow.keras.datasets import fashion_mnist

import os, json
import numpy as np 

import boto3
from botocore.exceptions import ClientError

def uploadFile(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def main():

    print('+------------------------------------------------')
    print('|  Downloading the training data              ')
    print('+------------------------------------------------')
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    config = json.load(open('config/awsConfig/awsConfig.json'))
    bucket = config['s3bucket']

    baseFolder = 'data/pipelines/rawData'
    os.makedirs(baseFolder, exist_ok=True)

    np.save( os.path.join(baseFolder, 'train_images.npy'), train_images )
    np.save( os.path.join(baseFolder, 'train_labels.npy'), train_labels )
    np.save( os.path.join(baseFolder, 'test_images.npy') , test_images  )
    np.save( os.path.join(baseFolder, 'test_labels.npy') , test_labels  )

    uploadFile( os.path.join(baseFolder, 'train_images.npy') , bucket, 'data/pipelines/rawData/train_images.npy' )
    uploadFile( os.path.join(baseFolder, 'train_labels.npy') , bucket, 'data/pipelines/rawData/train_labels.npy' )
    uploadFile( os.path.join(baseFolder, 'test_images.npy')  , bucket, 'data/pipelines/rawData/test_images.npy'  )
    uploadFile( os.path.join(baseFolder, 'test_labels.npy')  , bucket, 'data/pipelines/rawData/test_labels.npy'  )

    return

if __name__ == "__main__":
    main()
