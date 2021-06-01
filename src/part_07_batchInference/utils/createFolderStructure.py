import os, json, shutil
from distutils.dir_util import copy_tree
import tarfile
import boto3


def makeTarfile(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname = os.path.basename(source_dir))

def createFolder(config):

    if os.path.exists(config['newFolder']):
        print('folder already exists. Removing the folder')
        shutil.rmtree(config['newFolder'])

    print('creating the folder')
    os.makedirs(config['newFolder'], exist_ok=False)

    print('Copy the folder with the model:')
    first, last = os.path.split(config['modelFolder'])
    toFolder = os.path.join(config['newFolder'], last)
    os.makedirs(toFolder, exist_ok=False)
    copy_tree( config['modelFolder'] , toFolder)

    print('Copy the folder the inference code:')
    first, last = os.path.split(config['codeFolder'])
    toFolder = os.path.join(config['newFolder'], last)
    os.makedirs( toFolder )
    copy_tree( config['codeFolder'] , toFolder)

    return

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

    config    = json.load(open('src/part_07_batchInference/config.json'))
    
    print('-------------[Generating the model folder]---------------------')
    createFolder( config )

    print('-------------[Zipping to a .tar.gz file]---------------------')
    makeTarfile( config['newFolder'], config['tarFile'] )

    print('-------------[Uploading to S3]---------------------')
    uploadFile( config['tarFile'], config['bucket'], config['s3model'])

    print('-------------[Deleting the .tar.gz file]---------------------')
    os.unlink( config['tarFile'] )

    print('-------------[Deleting the new folder]---------------------')
    shutil.rmtree( config['newFolder'] )


if __name__ == "__main__":
    main()

