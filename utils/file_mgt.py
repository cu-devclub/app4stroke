import os
import re
import zipfile
import boto3


def download_data(data_path: str, save_path: str, client_configs: dict):
    
    """
    data_path (str) : path to bucket ex. 's3://bucket/dir/file'
    save_path (str) : path to write the downloaded file ex. '/home/user/app/temp/key/file'
    client_configs (dict) : dictionary of client access configurations 
                            ex. {
                                'aws_access_key_id': id, 
                                'aws_secret_access_key': secret,
                                'endpoint_url': url,
                                }
    """

    path_split = data_path.split('//')[1].split('/')
    bucket_name = path_split[0]
    object_name = '/'.join(path_split[1:])
    session = boto3.Session()
    client = session.client('s3', **client_configs)
    client.download_file(bucket_name, object_name, save_path)


def extract_nested_zip(zippedFile, toFolder):
    """ Extract a zip file including any nested zip files
        Delete the zip file(s) after extraction
        https://stackoverflow.com/a/43896058
    """
    with zipfile.ZipFile(zippedFile, 'r') as zfile:
        zfile.extractall(path=toFolder)
    os.remove(zippedFile)
    for root, _, files in os.walk(toFolder):
        for filename in files:
            if re.search(r'\.zip$', filename):
                fileSpec = os.path.join(root, filename)
                extract_nested_zip(fileSpec, root)


def list_all_files(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        if len(files) > 0:
            for f in files:
                file_paths.append(os.path.join(root, f))

    return file_paths
