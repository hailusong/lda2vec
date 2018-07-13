#!/bin/bash

python -u preprocess.py &> preprocess.log &
tail -f preprocess.log
