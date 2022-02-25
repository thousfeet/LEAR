devices=$1
checkpoint=$2
CUDA_VISIBLE_DEVICES=$devices
python main.py --mode test --checkpoint $checkpoint --task geo