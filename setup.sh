#!/bin/bash

if [ ! -f .venv/bin/activate ]; then
    python3 -m venv .venv --prompt BotRec
fi
source .venv/bin/activate
python3 -m pip install -U pip setuptools

pip install -U pyaudio numpy scipy
pip install -U matplotlib
