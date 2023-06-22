import argparse
import os
from typing import Any, Dict
from composer.utils import maybe_create_object_store_from_uri, parse_uri

def main(file_path: str, dataset_subset: str):
    """
    Main function to upload a JSONL file to a cloud storage.

    Args:
        file_path (str): The local path of the JSONL file to be uploaded.
        dataset_subset (str): The subset of the dataset to be uploaded.

    """
    # Create the object store to use for uploading data
    object_store = maybe_create_object_store_from_uri(file_path)
    _, _, cloud_folder_prefix = parse_uri(file_path)

    # Upload the file to the cloud storage
    object_store.upload_object(object_name=os.path.join(cloud_folder_prefix, dataset_subset),
                               filename=file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload JSONL data to cloud storage')
    parser.add_argument(
        '--folder_for_upload',
        type=str,
        help='The path of the JSONL file to upload')
    parser.add_argument(
        '--dataset_subset',
        type=str,
        default='small_full',
        help='Dataset subset to upload. Options are large_full and small_full')
    args = parser.parse_args()

    main(args.folder_for_upload, args.dataset_subset)
