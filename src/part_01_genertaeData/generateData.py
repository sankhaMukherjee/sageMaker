import tensorflow as tf
import sagemaker

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras          import utils

import os, json, logging
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

    X_train = (train_images/255).astype('float32')
    X_test = (test_images/255).astype('float32')

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # Convert to float32 numpy arrays
    numClasses = 10
    y_train = utils.to_categorical(train_labels, numClasses)
    y_test  = utils.to_categorical(test_labels, numClasses)

    print('\nGenerated array shapes:')
    print(f'X_train : {X_train.shape}')
    print(f'X_test  : {X_test.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'y_test  : {y_test.shape}')

    if not os.path.exists('data'):
        print('The ./data folder does not exist. Generating the ./data folder')
        os.makedirs('data')

    print('+------------------------------------------------')
    print('|  Saving the data to a local folder             ')
    print('+------------------------------------------------')

    np.save( 'data/X_train.npy', X_train )    
    np.save( 'data/y_train.npy', y_train )    
    np.save( 'data/X_test.npy', X_test )    
    np.save( 'data/y_test.npy', y_test )

    # np.savez('data/training', image=X_train, label=y_train)
    # np.savez('data/validation', image=X_test, label=y_test)

    print('\nGenerated data within the ./data folder:')
    print(os.listdir('data'))


    print('+------------------------------------------------')
    print('|  Saving the data to the S3 bucket              ')
    print('+------------------------------------------------')

    role   = json.load( open('config/awsConfig/awsConfig.json') )['arn']
    bucket = json.load( open('config/awsConfig/awsConfig.json') )['s3bucket']

    # sess   = sagemaker.Session()

    # s3Paths = []
    # s3Paths.append( sess.upload_data( 'data/X_train.npy', key_prefix = prefix + '/training' )   )
    # s3Paths.append( sess.upload_data( 'data/y_train.npy', key_prefix = prefix + '/training' )   )
    # s3Paths.append( sess.upload_data( 'data/X_test.npy',  key_prefix = prefix + '/validation' ) )
    # s3Paths.append( sess.upload_data( 'data/y_test.npy',  key_prefix = prefix + '/validation' ) )

    # print(f'Generated s3 paths:[{ ", ".join(s3Paths) }]')

    uploadFile( 'data/X_train.npy', bucket, 'training/X_train.npy' )
    uploadFile( 'data/y_train.npy', bucket, 'training/y_train.npy' )
    uploadFile( 'data/X_test.npy',  bucket, 'validation/X_test.npy' )
    uploadFile( 'data/y_test.npy',  bucket, 'validation/y_test.npy' )

    return

if __name__ == "__main__":
    main()
