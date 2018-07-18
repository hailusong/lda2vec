#!/bin/bash

python -u preprocess.py > preprocess.log 2>&1 &
tail -f preprocess.log
