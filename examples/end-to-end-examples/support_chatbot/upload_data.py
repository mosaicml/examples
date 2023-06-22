import argparse
import os
from typing import Any, Dict
from composer.utils import maybe_create_object_store_from_uri, parse_uri

def main(file_path: str, cloud_folder_for_upload: str):
    """
    Main function to upload a JSONL file to a cloud storage.

    Args:
        file_path (str): The local path of the JSONL file to be uploaded.
        cloud_folder_for_upload (str): The cloud storage URI where the file will be uploaded.

    """
    print(os.cwd())
    print(os.listdir())
    # Create the object store to use for uploading data
    object_store = maybe_create_object_store_from_uri(cloud_folder_for_upload)
    _, _, cloud_folder_prefix = parse_uri(cloud_folder_for_upload)

    # Upload the file to the cloud storage
    object_store.upload_object(object_name=os.path.join(cloud_folder_prefix, os.path.basename(file_path)),
                               filename=file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload JSONL data to cloud storage')
    parser.add_argument(
        '--local_file_path',
        type=str,
        help='The path of the local JSONL file to upload')
    parser.add_argument(
        '--cloud_folder_for_upload',
        type=str,
        help='The cloud storage URI where the file will be uploaded')
    args = parser.parse_args()

    main(args.local_file_path, args.cloud_folder_for_upload)
