# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List
import concurrent.futures

import datasets
from composer.utils import (ObjectStore, maybe_create_object_store_from_uri,
                            parse_uri)
from tqdm import tqdm


def dump_doc(doc_to_dump: Dict[str, Any], text_to_dump: List[str],
             object_store: ObjectStore, save_prefix: str):
    with TemporaryDirectory() as _tmp_dir:
        metadata = {
            'cik': doc_to_dump['cik'],
            'labels': doc_to_dump['labels'],
            'filingDate': doc_to_dump['filingDate'],
            'docID': doc_to_dump['docID'],
            'tickers': doc_to_dump['tickers'],
            'exchanges': doc_to_dump['exchanges'],
            'entityType': doc_to_dump['entityType'],
            'sic': doc_to_dump['sic'],
            'stateOfIncorporation': doc_to_dump['stateOfIncorporation'],
            'tickerCount': doc_to_dump['tickerCount'],
            'acceptanceDateTime': doc_to_dump['acceptanceDateTime'],
            'form': doc_to_dump['form'],
            'reportDate': doc_to_dump['reportDate'],
        }

        ticker_dir = Path(f'{metadata["tickers"][0]}')
        ticker_dir_full = _tmp_dir / ticker_dir
        os.makedirs(ticker_dir_full, exist_ok=True)

        year = metadata['reportDate'].split('-')[0]
        text_file_name = f'sec_{year}_txt.txt'
        metadata_file_name = f'sec_{year}_metadata.json'

        local_metadata_file_path = Path(ticker_dir_full) / Path(
            metadata_file_name)
        with open(local_metadata_file_path, 'w') as _json_file:
            json.dump(metadata, _json_file)

        local_text_file_path = Path(ticker_dir_full) / Path(text_file_name)
        with open(local_text_file_path, 'w') as _txt_file:
            for section in text_to_dump:
                _txt_file.write(section)
                _txt_file.write('\n\n')

        object_store.upload_object(object_name=os.path.join(
            save_prefix, ticker_dir, text_file_name),
                                   filename=local_text_file_path)
        object_store.upload_object(object_name=os.path.join(
            save_prefix, ticker_dir, metadata_file_name),
                                   filename=local_metadata_file_path)


def main(folder_for_upload: str, dataset_subset: str):
    object_store = maybe_create_object_store_from_uri(folder_for_upload)
    _, _, folder_prefix = parse_uri(folder_for_upload)

    batch_size = 10000

    for split in ['validation', 'test', 'train']:
        sub_prefix = os.path.join(folder_prefix, split)
        sec_filing_data = datasets.load_dataset(
            'JanosAudran/financial-reports-sec',
            dataset_subset,
            num_proc=os.cpu_count() - 2,
            split=split)
        sec_filing_data.remove_columns(['returns'])
        sorted_by_doc = sec_filing_data.sort(['docID', 'sentenceCount'])
        dataset_iter = sorted_by_doc.iter(batch_size=batch_size)

        docs_to_dump = []
        previous_doc = None
        running_text_sections = []
        running_text_section = []
        previous_section_id = None

        for packed_batch in tqdm(dataset_iter, total=math.ceil(len(sorted_by_doc) / batch_size), desc=f'Processing {split}'):
            unpacked_batch = []
            batch_length = len(packed_batch['docID'])
            for i in range(batch_length):
                unpacked_batch.append({
                    'cik': packed_batch['cik'][i],
                    'labels': packed_batch['labels'][i],
                    'filingDate': packed_batch['filingDate'][i],
                    'docID': packed_batch['docID'][i],
                    'tickers': packed_batch['tickers'][i],
                    'exchanges': packed_batch['exchanges'][i],
                    'entityType': packed_batch['entityType'][i],
                    'sic': packed_batch['sic'][i],
                    'stateOfIncorporation': packed_batch['stateOfIncorporation'][i],
                    'tickerCount': packed_batch['tickerCount'][i],
                    'acceptanceDateTime': packed_batch['acceptanceDateTime'][i],
                    'form': packed_batch['form'][i],
                    'reportDate': packed_batch['reportDate'][i],
                    'section': packed_batch['section'][i],
                    'sentence': packed_batch['sentence'][i]
                })

            for current_doc in unpacked_batch:
                if previous_doc is not None and current_doc[
                        'docID'] != previous_doc['docID']:
                    docs_to_dump.append((previous_doc, running_text_sections))

                    running_text_sections = []
                    running_text_section = []
                    previous_doc = current_doc
                    previous_section_id = current_doc['section']

                current_section_id = current_doc['section']
                if current_section_id != previous_section_id:
                    running_text_sections.append(' '.join(running_text_section))
                    running_text_section = []
                    previous_section_id = current_section_id

                running_text_section.append(current_doc['sentence'])

                previous_doc = current_doc

        running_text_sections.append(' '.join(running_text_section))
        docs_to_dump.append((previous_doc, running_text_sections))

        def dump_doc_wrapper(args):
            doc_to_dump, text_to_dump, object_store, sub_prefix = args
            dump_doc(doc_to_dump, text_to_dump, object_store, sub_prefix)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() - 2)) as executor:
            args = [(doc_to_dump, text_to_dump, object_store, sub_prefix) for doc_to_dump, text_to_dump in docs_to_dump]
            list(tqdm(executor.map(dump_doc_wrapper, args), total=len(args), desc=f'Uploading {split}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process and upload 10k data to object store')
    parser.add_argument('--folder_for_upload', type=str)
    parser.add_argument('--dataset_subset', type=str, default='small_full')
    args = parser.parse_args()

    main(args.folder_for_upload, args.dataset_subset)
