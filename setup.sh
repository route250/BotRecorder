#!/bin/bash

if [ ! -f .venv/bin/activate ]; then
    python3 -m venv .venv --prompt ELRec
fi
source .venv/bin/activate
python3 -m pip install -U pip setuptools

pip install -U pyaudio 'numpy<2.0.0' scipy
pip install -U matplotlib
