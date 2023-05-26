#!/bin/env bash

# Script to cd into a subdirectory, install its deps, and lint its contents
#
# The one subtlety here is that we only install the CPU dependencies since
# we don't have the CI infra to run workflows on GPUs yet. Also, it makes it
# easy to run this on your local machine + checks that we're doing
# conditional imports properly.

ENV_NAME="env-${1%/}"   # strip trailing slash if present

echo "Creating venv..."
python -m venv "$ENV_NAME" --system-site-packages
source "$ENV_NAME/bin/activate"

echo "Installing requirements..."
pip install --upgrade 'pip<23'
pip install 'pre-commit>=2.18.1,<3'
pip install 'pyright==1.1.296'
pip install 'pytest>=7.2.1,<8'
target=$(echo $1 | tr '_' '-' | tr '/' '-')

original_dir=$(pwd)
cd examples/$1
if [ -f requirements-cpu.txt ]; then
    pip install -r requirements-cpu.txt
elif [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "No requirements-cpu.txt or requirements.txt found in directory examples/$1"
fi
cd $original_dir


echo "Running checks on files:"
FILES=$(find "examples/$1" -type f | grep -v '.pyc')
echo $FILES
PYTHON_FILES=$(echo "$FILES" | grep '\.py$')
pre-commit run --files $FILES && ([ -z "$PYTHON_FILES" ] || pyright $PYTHON_FILES)
STATUS=$?

echo "Cleaning up venv..."
deactivate
rm -rf "$ENV_NAME"

exit $STATUS
