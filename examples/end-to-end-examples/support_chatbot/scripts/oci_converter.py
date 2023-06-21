from streaming import StreamingDataset
from torch.utils.data import DataLoader
from getpass import getpass
from tqdm import tqdm
from urllib.parse import urlparse
import os
import hashlib

MOSAICML_API_TOKEN = getpass()
DATA_PATH = 'examples/end-to-end-examples/support_chatbot/retrieval_data/pypi_documentation'

MAX_ENTRIES_PER_FILE = 5000  # Maximum number of entries per file

def main():
    remote_dir = 'oci://mosaicml-internal-datasets/mpt-swe-filtered'
    local_dir = 'local_dir'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=False)
    dataloader = DataLoader(dataset)
    os.makedirs(DATA_PATH, exist_ok=True)

    current_file_entries = 0  # Counter for the number of entries in the current file
    file_count = 0  # Counter for the number of files

    # Create the initial file
    file_path = os.path.join(DATA_PATH, f'data_{file_count}.txt')
    file = open(file_path, 'w')

    for doc_data in tqdm(dataloader, desc='Loading PyPi', total=len(dataloader)):
        docstring = doc_data['docstring']
        docstring = ', '.join(docstring)
        file_name = doc_data['file_name']
        file_name = ', '.join(file_name)
        package_name = doc_data['package_name']
        package_name = ', '.join(package_name)
        source_code = doc_data['source_code']
        source_code = ', '.join(source_code)
        url = doc_data['url']
        url = ', '.join(url)

        # Create the entry
        entry = 'package: ' + package_name + ', ' + 'file: ' + file_name + '\n' + \
                docstring + '\n' + \
                source_code + '\n'
                

        # If this entry makes the file exceed the maximum number of entries, create a new file
        if current_file_entries == MAX_ENTRIES_PER_FILE:
            file.close()
            file_count += 1
            file_path = os.path.join(DATA_PATH, f'batch_{file_count}.txt')
            file = open(file_path, 'w', encoding='utf-8')
            current_file_entries = 0

        # Write the entry to the file
        file.write(entry)
        current_file_entries += 1

    # Close the last file
    file.close()

if __name__ == '__main__':
    main()
