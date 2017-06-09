#!/bin/bash
source activate pytorch35
python test.py --test_file data/senti.binary.test.txt --model_file model_hw1.th --learning_rate 0.001 --decompose_type beta --heatmap_file test_heatmap_beta.html


