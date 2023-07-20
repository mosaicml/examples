from streaming import StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import random

from transformers import AutoTokenizer

def check_raw():
    print('------------------------ Starting check ------------------------')
    remote_source_folders = ['source_code_processed_train', 'source_code_processed_val']
    for folder in remote_source_folders:
        remote_dir = os.path.join('oci://mosaicml-internal-datasets/mpt-swe-filtered/', folder)
        dataset = StreamingDataset(remote=remote_dir, split=None, shuffle=False)
        dataloader = DataLoader(dataset)

        url_hash = set()
        dup_count = 0
        store = []
        for doc_data in tqdm(dataloader, desc='Loading PyPi docstrings', total=len(dataloader)):
            text = doc_data['text']
            text = ', '.join(text)
            url = doc_data['url']
            url = ', '.join(url)
            package_name = doc_data['package_name']
            package_name = ', '.join(package_name)

            if f'{url}__{package_name}__{text}' in url_hash:
                print(f'duplicate in {folder}: {url}__{package_name}')
                dup_count += 1
            else:
                url_hash.add(f'{url}__{package_name}__{text}')
        store.append((folder, dup_count, len(dataloader)))
    print('------------------------ Completed check ------------------------')
    print(store)


def main() -> None:
    '''
    for d_set in ['train', 'validation']:
        tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
        dataset = StreamingDataset(remote=f'oci://mosaicml-internal-checkpoints/support-bot-demo/data/PyPi_mds/{d_set}', split=None, shuffle=False)
        dataloader = DataLoader(dataset)

        random_numbers = set()
        for _ in range(5):
            random_numbers.add(random.randint(0, len(dataloader)-1))

        token_count = 0
        with open(f'myfile_{d_set}.txt', 'w') as f:
            for doc_data in tqdm(dataloader, desc='Loading PyPi', total=len(dataloader)):
                if token_count in random_numbers:
                    tmp = ''
                    f.write(f'concat_tok_num: {token_count}---------------------\n')
                    for i in doc_data['tokens']:
                        arr = np.frombuffer(i, dtype=np.uint32)
                        # Write the string to the file
                        tmp += f'{tokenizer.decode(arr, skip_special_tokens=True)}'
                    f.write(f'{tmp}\n')
                token_count += 1
        print(f'{token_count}')
    '''
    check_raw()


if __name__ == '__main__':
    main()