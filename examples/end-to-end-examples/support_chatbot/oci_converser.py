import oci
import os

class OCIObjectStorageManager:
    def __init__(self, 
                 oci_uri: str,
                 config_path="~/.oci/config", 
                 profile_name="DEFAULT"):
        self.config = oci.config.from_file(config_path, profile_name)
        self.object_storage_client = oci.object_storage.ObjectStorageClient(self.config)
        self.bucket_name, self.path = self.parse_oci_uri(oci_uri)
        self.namespace = self._get_namespace()

    def parse_oci_uri(self, oci_uri):
        """
        Parse an OCI URI into its components.

        Args:
        - oci_uri (str): The OCI URI to parse.

        Returns:
        - tuple: A tuple containing the bucket name and object path.
        """
        if not oci_uri.startswith("oci://"):
            raise ValueError(f"Invalid OCI URI: {oci_uri}")

        # Remove the "oci://" prefix
        stripped = oci_uri[6:]

        # Split into bucket name and object path
        parts = stripped.split("/", 1)
        bucket_name = parts[0]

        object_path = None
        if len(parts) > 1:
            object_path = parts[1]

        return bucket_name, object_path

    def _get_namespace(self):
        """Fetch the Object Storage namespace."""
        return self.object_storage_client.get_namespace().data

    def _get_oci_object_name(self, filename):
        """Get the object name with oci_path."""
        return os.path.join(self.path, filename)

    def upload_file_obj(self, file_obj, object_name):
        """Upload a BytesIO object to an OCI bucket."""
        object_name = self._get_oci_object_name(object_name)
        print(f"Uploading {object_name}...")
        self.object_storage_client.put_object(
            self.namespace,
            self.bucket_name,
            object_name,
            file_obj
        )
        print(f"Uploaded {object_name} to {self.bucket_name}/{object_name}.")

    def upload_directory(self, local_path):
        """Upload all files from a local directory to an OCI bucket."""
        for filename in os.listdir(local_path):
            filepath = os.path.join(local_path, filename)
            with open(filepath, 'rb') as file_obj:
                self.upload_file_obj(file_obj, filename, self.bucket_name)

    def download_directory(self, local_path):
        """Download all files from an OCI bucket to a local directory."""
        prefix = self.path
        objects = self.object_storage_client.list_objects(self.namespace, self.bucket_name, prefix=prefix)
        
        for obj in objects.data.objects:
            self.download_file(local_path, obj.name)

    def download_file(self, local_path, object_name):
        """Download a single file from an OCI bucket to a local directory."""
        print(f"Downloading {object_name}...")
        
        response = self.object_storage_client.get_object(self.namespace, self.bucket_name, object_name)
        local_filename = os.path.basename(object_name)
        with open(os.path.join(local_path, local_filename), 'wb') as file:
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                file.write(chunk)

        print(f"Downloaded {object_name} to {local_path}.")

if __name__ == '__main__':
    # Edit these values accordingly
    local_path = "/path/to/local/directory"  # Path to local directory
    oci_path = "YOUR_OCI_PATH"  # Folder or prefix in OCI
    
    manager = OCIObjectStorageManager(oci_path)
    manager.upload_directory(local_path)
    manager.download_directory(local_path)