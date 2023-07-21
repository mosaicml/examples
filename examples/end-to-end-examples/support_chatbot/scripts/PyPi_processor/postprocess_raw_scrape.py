
## scrape_pypi.analyze_scrape
import random
from textwrap import wrap
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from transformers import AutoTokenizer
import sys
import numpy as np
import tqdm
import os
import shutil
import re
from streaming import MDSWriter

interesting_extensions = set(
    [
    'py',
    ]
)
if __name__ == "__main__":
    args = sys.argv[1:]
    if os.path.exists('local_dir'):
        shutil.rmtree('local_dir')
    # Remote directory (S3 or local filesystem) where dataset is stored
    remote_dir = 'oci://mosaicml-internal-datasets/pypi/docstring_functions'

    # Local directory where dataset is cached during operation
    local_dir = 'local_dir'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=False)

    # Create PyTorch DataLoader
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
    dataloader = DataLoader(dataset)
    stats = {
        'pairs': 0,
        'source_tokens': 0,
        'docstring_tokens': 0
    }
    blacklisted_files = {}
    for batch in tqdm.tqdm(dataloader):
        if batch['package_name'][0] not in blacklisted_files:
            blacklisted_files[batch['package_name'][0]] = {}

        if batch['file_name'][0] not in blacklisted_files[batch['package_name'][0]]:
            blacklisted_files[batch['package_name'][0]][batch['file_name'][0]] = []
        blacklisted_files[batch['package_name'][0]][batch['file_name'][0]].append(
            (f'{batch["docstring"][0]}', batch['source_code'][0])
        )
        stats['pairs'] += 1
    
    print(len(blacklisted_files))
    print(stats)

    if os.path.exists('local_dir_1'):
        shutil.rmtree('local_dir_1')

    # Remote directory (S3 or local filesystem) where dataset is stored
    remote_dir = 'oci://mosaicml-internal-datasets/pypi/source_code'

    # Local directory where dataset is cached during operation
    local_dir = 'local_dir_1'
    dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=False)

    # Create PyTorch DataLoader
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
    dataloader = DataLoader(dataset)
  
    most_recent_pkg_name = None
    concatenated_project_data = {
        'url': None,
        'package_name': None,
        'source_funcs': [],
        'docstrings': [],
    }
    MAX_SEQ_LEN = 2048
    curr_text = ""


    doc_train_s3 = "oci://mosaicml-internal-checkpoints/support-bot-demo/data/mpt-swe-2k/docstring_processed_train/"
    doc_val_s3 = "oci://mosaicml-internal-checkpoints/support-bot-demo/data/mpt-swe-2k/docstring_processed_val/"
    doc_test_s3 = "oci://mosaicml-internal-checkpoints/support-bot-demo/data/mpt-swe-2k/docstring_processed_test/"

    doc_columns = {
        "instruction": "str",
        "completion": "str",
        "url": "str",
        "package_name": "str",
        "file_name": "str"
    }
    src_train_s3 = "oci://mosaicml-internal-checkpoints/support-bot-demo/data/mpt-swe-2k/source_code_processed_train/"
    src_val_s3 = "oci://mosaicml-internal-checkpoints/support-bot-demo/data/mpt-swe-2k/source_code_processed_val/"
    src_columns = {
        "text": "str",
        "url": "str",
        "package_name": "str",
        "file_name": "str"
    }

  
    compression = 'zstd:16'
    hashes = ['sha1', 'xxh64']
    size_limit = 134217728

    dup_hash = set()

    with MDSWriter(out=('docs_train_src', doc_train_s3), columns=doc_columns, keep_local=False, compression=compression, hashes=hashes, size_limit=size_limit) as doc_train_out:
        with MDSWriter(out=('docs_val_src', doc_val_s3), columns=doc_columns, keep_local=False, compression=compression, hashes=hashes, size_limit=size_limit) as doc_val_out:
            with MDSWriter(out=('docs_test_src', doc_test_s3), columns=doc_columns, keep_local=False, compression=compression, hashes=hashes, size_limit=size_limit) as doc_test_out:
                with MDSWriter(out=('src_train', src_train_s3), columns=src_columns, keep_local=False, compression=compression, hashes=hashes, size_limit=size_limit) as src_train_out:
                    with MDSWriter(out=('src_val', src_val_s3), columns=src_columns, keep_local=False, compression=compression, hashes=hashes, size_limit=size_limit) as src_val_out:
                        idx = 0
                        for batch in tqdm.tqdm(dataloader):
                            idx += 1
                            print(f"{idx}", flush=True)
                            
                            if most_recent_pkg_name != batch['package_name'][0]:
                                
                                if concatenated_project_data['url'] is not None:
                                    sampl = random.random()
                                    if sampl < 0.1:
                                        src_val_out.write({
                                            "text": curr_text,
                                            "url": concatenated_project_data['url'],
                                            "package_name":  concatenated_project_data['package_name'],
                                            "file_name": batch['file_name'][0].split('/')[-1]
                                        })
                                    else:
                                        src_train_out.write({
                                            "text": curr_text,
                                            "url": concatenated_project_data['url'],
                                            "package_name":  concatenated_project_data['package_name'],
                                            "file_name": batch['file_name'][0].split('/')[-1]
                                        }) 
                                
                                
                                concatenated_project_data = {
                                    'url': None,
                                    'package_name': None,
                                    'source_funcs': [],
                                    'docstrings': [],
                                }  
                                # new package
                                most_recent_pkg_name = batch['package_name'][0]
                                curr_text = f"### Package: {batch['package_name'][0]} ###"
                            
                            concatenated_project_data['url'] = batch['url'][0]
                            concatenated_project_data['package_name'] = batch['package_name'][0]

                            fn = batch['file_name'][0].split('/')[-1]
                            if len(fn.split('.')) == 1:
                                extension = 'none'
                            else:
                                extension = fn.split('.')[-1].lower()
                            if extension not in interesting_extensions and extension != 'none':
                                continue
                            
                            for i in range(0, len(batch['text'][0]), 250_000):
                                file_text_chunk = batch['text'][0][i * 250_000 : (i+1) * 250_000]
                                if batch['package_name'][0] in blacklisted_files and batch['file_name'][0] in blacklisted_files[batch['package_name'][0]]:
                                    for docstring, source_func in blacklisted_files[batch['package_name'][0]][batch['file_name'][0]]:
                                        func_lst = source_func.split('\n')
                                        begin, end = func_lst[0], func_lst[-1]
                                        
                                        if begin not in file_text_chunk:
                                            concatenated_project_data['source_funcs'].append(source_func)
                                            concatenated_project_data['docstrings'].append(docstring)
                                            continue
                                        start_idx = file_text_chunk.index(begin)
                                        if end not in file_text_chunk[start_idx+1:]:
                                            end_idx = len(file_text_chunk)
                                        else:
                                            end_idx = file_text_chunk.index(end, start_idx+1) + len(end)
                                        file_text_chunk = file_text_chunk[:start_idx] +file_text_chunk[end_idx:]

                                        
                                        concatenated_project_data['source_funcs'].append(source_func)
                                        concatenated_project_data['docstrings'].append(docstring)

                                
                                
                                if len(tokenizer(f"{curr_text}\n{file_text_chunk}")['input_ids']) > MAX_SEQ_LEN:
                                    sampl = random.random()
                                    if sampl < 0.1:
                                        src_val_out.write({
                                            "text": curr_text,
                                            "url": concatenated_project_data['url'],
                                            "file_name": batch['file_name'][0].split('/')[-1],
                                            "package_name":  concatenated_project_data['package_name']
                                        })
                                    else:
                                        src_train_out.write({
                                            "text": curr_text,
                                            "url": concatenated_project_data['url'],
                                            "file_name": batch['file_name'][0].split('/')[-1],
                                            "package_name":  concatenated_project_data['package_name']
                                        })
                                    curr_text = f"### Package: {batch['package_name'][0]} ###"
                                
                                curr_text += '\n' + file_text_chunk

                                
                                
                                if batch['package_name'][0] in blacklisted_files and batch['file_name'][0] in blacklisted_files[batch['package_name'][0]]:
                                    for docstring, source_func in blacklisted_files[batch['package_name'][0]][batch['file_name'][0]]:
                                        matches = re.findall('([a-z_]+)', source_func.split('\n')[0])
                                        matches = list(filter(lambda x: x !='def', matches))
                                        if len(matches) == 0:
                                            func_name = 'foo'
                                        else:
                                            func_name = matches[-1]
                                        
                                        if random.random() < 0.5:
                                            instruction = f'Reference source code:\n{curr_text}\n### Code generation task ###\nUsing the python package source code above as a reference, write a python function named {func_name} that matches the following docstring.\nDocstring:\n"""\n{docstring}\n"""\nSource code:\n'
                                            desired_output = source_func
                                        else:
                                            instruction = f'Reference source code:\n{curr_text}\n### Docstring writing task ###\nUsing the python package source code above as a reference, write a docstring that describes the function below.\nSource code:\n{source_func}\nDocstring:\n'
                                            desired_output = docstring
                                        sampl = random.random()
                                        if sampl < 0.1:
                                            doc_test_out.write({
                                                "instruction": instruction,
                                                "completion": desired_output,
                                                "url": concatenated_project_data['url'],
                                                "file_name": batch['file_name'][0].split('/')[-1],
                                                "package_name":  concatenated_project_data['package_name']
                                            })
                                        elif sampl > 0.1 and sampl <= 0.2:
                                            doc_val_out.write({
                                                "instruction": instruction,
                                                "completion": desired_output,
                                                "url": concatenated_project_data['url'],
                                                "file_name": batch['file_name'][0].split('/')[-1],
                                                "package_name":  concatenated_project_data['package_name']
                                            })
                                        elif sampl > 0.2:
                                            doc_train_out.write({
                                                "instruction": instruction,
                                                "completion": desired_output,
                                                "url": concatenated_project_data['url'],
                                                "file_name": batch['file_name'][0].split('/')[-1],
                                                "package_name":  concatenated_project_data['package_name']
                                            })
