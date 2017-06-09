#!/bin/bash
source activate pytorch35
python main.py --train_file data/train.json --dev_file data/dev.json --test_file data/test.json --model_file model_debug.th --learning_rate 0.001


