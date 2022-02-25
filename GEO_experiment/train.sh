devices=$1
checkpoint=$2
random_seed=$3
CUDA_VISIBLE_DEVICES=$devices python main.py --mode train --checkpoint $checkpoint --task geo --random-seed $random_seed