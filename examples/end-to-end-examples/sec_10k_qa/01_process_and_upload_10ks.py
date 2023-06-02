# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import datasets
from composer.utils.object_store import OCIObjectStore


def dump_doc(doc_to_dump, text_to_dump, object_store, save_prefix: str):
    with TemporaryDirectory() as _tmp_dir:
        doc_id = doc_to_dump['docID']

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
            'returns': doc_to_dump['returns'],
        }

        ticker_dir = Path(f'{metadata["tickers"][0]}')
        ticker_dir_full = _tmp_dir / ticker_dir
        os.makedirs(ticker_dir_full, exist_ok=True)

        text_file_name = f'sec_{doc_id}_txt.txt'
        metadata_file_name = f'sec_{doc_id}_metadata.json'

        local_metadata_file_path = Path(ticker_dir_full) / Path(
            metadata_file_name)
        with open(local_metadata_file_path, 'w') as _json_file:
            json.dump(metadata, _json_file)

        local_text_file_path = Path(ticker_dir_full) / Path(text_file_name)
        with open(local_text_file_path, 'w') as _txt_file:
            for section in text_to_dump:
                _txt_file.write(section)
                _txt_file.write('\n\n')

        object_store.upload_object(
            object_name=f'{save_prefix}/{ticker_dir}/{text_file_name}',
            filename=local_text_file_path)
        object_store.upload_object(
            object_name=f'{save_prefix}/{ticker_dir}/{metadata_file_name}',
            filename=local_metadata_file_path)


def main():
    oci_os = OCIObjectStore('mosaicml-internal-checkpoints-shared')
    prefix = 'daniel/data/sec-filings-small-train'

    sec_filing_data = datasets.load_dataset('JanosAudran/financial-reports-sec',
                                            'small_full',
                                            num_proc=os.cpu_count() - 2,
                                            split='train')
    sorted_by_doc = sec_filing_data.sort(['docID', 'sentenceCount'])

    previous_doc = None
    running_text_sections = []
    running_text_section = []
    previous_section_id = None
    for doc_index in range(len(sorted_by_doc)):
        current_doc = sorted_by_doc[doc_index]

        if current_doc['docID'] != previous_doc['docID']:
            if previous_doc is None:
                continue

            text = ' '.join(running_text_sections)
            dump_doc(previous_doc, text, oci_os, prefix)

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

    running_text_sections.append(' '.join(running_text_section))
    text = ' '.join(running_text_sections)
    dump_doc(previous_doc, text, oci_os, prefix)

    # with TemporaryDirectory() as _tmp_dir:
    #     running_doc_id = sorted_by_doc[0]['docID']
    #     running_section_id = sorted_by_doc[0]['section']
    #     doc_index = 0
    #     text_section_buffer = []
    #     text_doc_buffer = []
    #     metadata_buffer = None
    #     while doc_index < len(sorted_by_doc):
    #         if doc_index % 10000 == 0:
    #             print(f'{doc_index} out of {len(sorted_by_doc)}')
    #         current_doc = sorted_by_doc[doc_index]

    #         if metadata_buffer is None:
    #             metadata_buffer = {
    #                 'cik': current_doc['cik'],
    #                 'labels': current_doc['labels'],
    #                 'filingDate': current_doc['filingDate'],
    #                 'docID': current_doc['docID'],
    #                 'tickers': current_doc['tickers'],
    #                 'exchanges': current_doc['exchanges'],
    #                 'entityType': current_doc['entityType'],
    #                 'sic': current_doc['sic'],
    #                 'stateOfIncorporation': current_doc['stateOfIncorporation'],
    #                 'tickerCount': current_doc['tickerCount'],
    #                 'acceptanceDateTime': current_doc['acceptanceDateTime'],
    #                 'form': current_doc['form'],
    #                 'reportDate': current_doc['reportDate'],
    #                 'returns': current_doc['returns'],
    #             }

    #         if current_doc['docID'] != running_doc_id:
    #             text_doc_buffer.append(' '.join(text_section_buffer))
    #             text_section_buffer = []

    #             ticker_dir = Path(f'{metadata_buffer["tickers"][0]}')
    #             ticker_dir_full = _tmp_dir / ticker_dir
    #             os.makedirs(ticker_dir_full, exist_ok=True)

    #             text_file_name = f'sec_{running_doc_id}_txt.txt'
    #             metadata_file_name = f'sec_{running_doc_id}_metadata.json'

    #             local_metadata_file_path = Path(ticker_dir_full) / Path(metadata_file_name)
    #             with open(local_metadata_file_path, 'w') as _json_file:
    #                 json.dump(metadata_buffer, _json_file)

    #             local_text_file_path = Path(ticker_dir_full) / Path(text_file_name)
    #             with open(local_text_file_path, 'w') as _txt_file:
    #                 for section in text_doc_buffer:
    #                     _txt_file.write(section)
    #                     _txt_file.write('\n\n')

    #             s3.upload_object(object_name=f'daniel/{prefix}/{ticker_dir}/{text_file_name}', filename=local_text_file_path)
    #             s3.upload_object(object_name=f'daniel/{prefix}/{ticker_dir}/{metadata_file_name}', filename=local_metadata_file_path)

    #             text_doc_buffer = []
    #             running_section_id = current_doc['section']
    #             running_doc_id = current_doc['docID']

    #             metadata_buffer = {
    #                 'cik': current_doc['cik'],
    #                 'labels': current_doc['labels'],
    #                 'filingDate': current_doc['filingDate'],
    #                 'docID': current_doc['docID'],
    #                 'tickers': current_doc['tickers'],
    #                 'exchanges': current_doc['exchanges'],
    #                 'entityType': current_doc['entityType'],
    #                 'sic': current_doc['sic'],
    #                 'stateOfIncorporation': current_doc['stateOfIncorporation'],
    #                 'tickerCount': current_doc['tickerCount'],
    #                 'acceptanceDateTime': current_doc['acceptanceDateTime'],
    #                 'form': current_doc['form'],
    #                 'reportDate': current_doc['reportDate'],
    #                 'returns': current_doc['returns'],
    #             }

    #         current_section_id = current_doc['section']
    #         if current_section_id != running_section_id:
    #             text_doc_buffer.append(' '.join(text_section_buffer))
    #             text_section_buffer = []
    #             running_section_id = current_section_id

    #         text_section_buffer.append(current_doc['sentence'])

    #         doc_index += 1

    #     text_doc_buffer.append(' '.join(text_section_buffer))
    #     text_section_buffer = []

    #     ticker_dir = Path(f'{current_doc["tickers"][0]}')
    #     ticker_dir_full = _tmp_dir / ticker_dir
    #     os.makedirs(ticker_dir_full, exist_ok=True)

    #     text_file_name = f'sec_{running_doc_id}_txt.txt'
    #     metadata_file_name = f'sec_{running_doc_id}_metadata.json'

    #     local_metadata_file_path = Path(ticker_dir_full) / Path(metadata_file_name)
    #     with open(local_metadata_file_path, 'w') as _json_file:
    #         json.dump(metadata_buffer, _json_file)

    #     local_text_file_path = Path(ticker_dir_full) / Path(text_file_name)
    #     with open(local_text_file_path, 'w') as _txt_file:
    #         for section in text_doc_buffer:
    #             _txt_file.write(section)
    #             _txt_file.write('\n')

    #     s3.upload_object(object_name=f'daniel/{prefix}/{ticker_dir}/{text_file_name}', filename=local_text_file_path)
    #     s3.upload_object(object_name=f'daniel/{prefix}/{ticker_dir}/{metadata_file_name}', filename=local_metadata_file_path)

    #     text_doc_buffer = []
    #     running_section_id = current_doc['section']
    #     running_doc_id = current_doc['docID']


if __name__ == '__main__':
    main()
