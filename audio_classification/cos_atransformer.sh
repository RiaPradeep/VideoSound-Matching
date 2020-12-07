#!/bin/sh
#SBATCH -o cos_audio_transformer.out # STDOUT
python train_cos.py --model audio_transformer