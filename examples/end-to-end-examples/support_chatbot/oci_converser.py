import oci
import os

class OCIObjectStorageManager:
    def __init__(self, config_path=".oci/config", profile_name="DEFAULT"):
        self.config = oci.config.from_file(config_path, profile_name)
        self.object_storage_client = oci.object_storage.ObjectStorageClient(self.config)

    def upload_directory(self, local_path, bucket_name, namespace_name):
        """Upload all files in a directory to an OCI bucket."""
        for root, dirs, files in os.walk(local_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                self.upload_file(file_path, bucket_name, namespace_name)

    def upload_file(self, file_path, bucket_name, namespace_name):
        """Upload a single file to an OCI bucket."""
        print(f"Uploading {file_path}...")
        
        with open(file_path, 'rb') as file:
            object_name = os.path.basename(file_path)
            self.object_storage_client.put_object(
                namespace_name,
                bucket_name,
                object_name,
                file
            )

        print(f"Uploaded {file_path} to {bucket_name}/{object_name}.")

    def download_directory(self, local_path, bucket_name, namespace_name):
        """Download all files from an OCI bucket to a local directory."""
        objects = self.object_storage_client.list_objects(namespace_name, bucket_name)
        
        for obj in objects.data.objects:
            self.download_file(local_path, obj.name, bucket_name, namespace_name)

    def download_file(self, local_path, object_name, bucket_name, namespace_name):
        """Download a single file from an OCI bucket to a local directory."""
        print(f"Downloading {object_name}...")
        
        response = self.object_storage_client.get_object(namespace_name, bucket_name, object_name)
        with open(os.path.join(local_path, object_name), 'wb') as file:
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                file.write(chunk)

        print(f"Downloaded {object_name} to {local_path}.")

if __name__ == '__main__':
    # Edit these values accordingly
    local_path = "/path/to/local/directory"  # Path to local directory
    bucket_name = "YOUR_BUCKET_NAME"
    namespace_name = "YOUR_NAMESPACE_NAME"  # e.g., tenancy name
    
    manager = OCIObjectStorageManager()

    # To upload
    manager.upload_directory(local_path, bucket_name, namespace_name)

    # To download
    manager.download_directory(local_path, bucket_name, namespace_name)
