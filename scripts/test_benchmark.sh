#!/bin/env bash

# Script to cd into a subdirectory, install its deps, and run its tests
#
# The tricky part here is that some modules, like flash-attn, are really
# finicky to get installed and depend on details of your environment (e.g.,
# the installed CUDA version). For these deps, we just use whatever's
# in the surrounding environment. We do this by:
#   1) Allowing the venv to use the environment's installed packages
#       via the --system-site-packages flag.
#   2) Having the pip install overwrite everything in the requirements
#       *except* a few whitelisted dependencies.

ENV_NAME="${1%/}-env"   # strip trailing slash if present

cd "$1"

# if [ "$2" ]; then
echo "Creating venv..."
python -m venv "$ENV_NAME" --system-site-packages
source "$ENV_NAME/bin/activate"
# fi

# # work around flash-attn needing torch already installed in its setup.py
# cat requirements.txt | grep 'torch' | xargs pip install

cat requirements.txt | grep -v 'flash-attn' > /tmp/requirements.txt
# cat requirements.txt | grep -Ev 'flash-attn|mm' > /tmp/requirements.txt

echo "Installing requirements:"
cat /tmp/requirements.txt
# cd -; exit
# pip install -I -r /tmp/requirements.txt
pip install -U -r /tmp/requirements.txt
rm /tmp/requirements.txt

# pip install -U -r requirements.txt

pip show flash-attn
python -c 'import torch; print(torch.__path__[0])'
python -c 'import numpy; print(numpy.__path__[0])'
python -c 'import flash_attn; print(flash_attn.__path__[0])'
# exit


pytest tests/*

# if [ "$2" ]; then
echo "Cleaning up venv..."
deactivate
rm -rf "$ENV_NAME"
# fi

cd -
