#!/bin/sh
#SBATCH -o video_avg_cnn.out # STDOUT
python train_cos.py --model video_avg_cnn