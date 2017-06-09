#!/bin/bash
source activate pytorch35
python main.py --train_file data/senti.binary.train.txt --dev_file data/senti.binary.dev.txt --test_file data/senti.binary.test.txt --model_file model_hw1.th --learning_rate 0.001

