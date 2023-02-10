#!/bin/bash
source venv-llm/bin/activate
composer train_clm/train_clm.py train_clm/yamls/gpt2/125m.yaml train_loader.dataset.split=train_small \
    eval_interval=0 \
    max_duration=3ba \
    save_interval=1ba save_folder="./mosaic-gpt-125m/"
