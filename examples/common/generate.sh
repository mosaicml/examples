#!/bin/bash
ROOT=/home/mosaicml/gpt2_tokenized_2k; python convert_c4.py --out_root $ROOT --splits train train_small val val_small --concat_tokens=2048 --tokenizer=gpt2 --eos_text="<|endoftext|>" --compression zstd
