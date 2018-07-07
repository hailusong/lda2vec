#!/bin/bash

python -u lda2vec_run.py &> lda2vec_run.log &
tail -f lda2vec_run.log
