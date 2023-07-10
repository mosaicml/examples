import oci
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Upload a file to OCI')
    parser.add_argument('--in_root', type=str, required=True,
                        help='The path to the source file')
    parser.add_argument('--out_root', type=str, required=True,
                        help='The destination object name in OCI')
    parser.add_argument('--namespace', type=str, required=True,
                        help='The namespace in OCI')
    parser.add_argument('--bucket_name', type=str, required=True,
                        help='The bucket name in OCI')
    return parser.parse_args()

def main(args: argparse.Namespace):
    config = oci.config.from_file() 

    object_storage_client = oci.object_storage.ObjectStorageClient(config)

    namespace = args.namespace
    bucket_name = args.bucket_name

    object_name = args.out_root  # destination object name in OCI
    file_path = args.in_root  # source file path

    with open(file_path, 'rb') as f:
        object_storage_client.put_object(namespace, bucket_name, object_name, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
