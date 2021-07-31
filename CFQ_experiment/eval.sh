dos2unix eval.sh

CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --mode test \
    --checkpoint test_mcd1 \
    --lesson 30 \
    --random-seed 4 \
    --data-path ./data/mcd1 \
    --pretrained-model ./checkpoint/models/mcd1/model.mdl 