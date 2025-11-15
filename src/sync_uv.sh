#!/bin/bash

rm -rf .venv
rm -rf uv.lock
rm -rf .python-version

uv python pin 3.7
uv run python -c "import urllib.request as u; u.urlretrieve('https://bootstrap.pypa.io/pip/3.7/get-pip.py', 'get-pip.py')"
uv run python get-pip.py
uv run python -m pip install "setuptools<66" wheel==0.38.4
uv run python -m pip install -r requirements.txt 
uv run python -m pip install compiler-gym
uv run python -m pip install torch

rm -rf get-pip.py
