#!/bin/bash
source venv-llm/bin/activate
composer train_clm/train_clm.py train_clm/yamls/opt/125m.yaml \
    train_loader.dataset.split=train_small \
    # data_local=/tmp/the-pile \
    # data_remote=s3://mosaicml-internal-dataset-the-pile/mds/2 \
    eval_interval=0 \
    max_duration=3ba \
    # save_interval=1ba save_folder="./opt-125m/"
