#!/bin/sh
#SBATCH -o cos_vcnn_lstm.out # STDOUT
python train_cos.py --model video_cnn_lstm