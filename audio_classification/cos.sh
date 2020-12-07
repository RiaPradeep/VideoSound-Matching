#!/bin/sh
#SBATCH -o cos_cnn_encoder.out # STDOUT
python train_cos.py --model cnn_encoder