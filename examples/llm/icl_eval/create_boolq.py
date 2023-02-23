# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys


def main(input_path, output_path):
    loaded_boolq = [
        json.loads(line) for line in open(input_path, 'r').readlines()
    ]
    output_boolq = []

    for row in loaded_boolq:
        output_boolq.append({
            'context': row['passage'] + ' Question: ' + row['question'] + '?',
            'continuation': 'yes' if row['answer'] else 'no',
        })

    with open(output_path, 'w') as f:
        for row in output_boolq:
            f.write(json.dumps(row))
            f.write('\n')

    print(len(output_boolq))


if __name__ == '__main__':
    # can be downloaded from https://github.com/google-research-datasets/boolean-questions
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    main(input_path, output_path)
