#!/bin/bash
source /workdisk/abhi/examples/venv-new/bin/activate
composer main.py yamls/mosaic_gpt/125m.yaml train_loader.dataset.split=train_small
