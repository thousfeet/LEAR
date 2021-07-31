dos2unix train.sh

  CUDA_VISIBLE_DEVICES=0 \
  nohup python main.py \
      --mode train \
      --checkpoint mcd1_lesson11 \
      --lesson 11 \
      --random-seed 4 \
      --data-path ./data/mcd1 \
      #--pretrained-model ./checkpoint/models/mcd1_lesson11/xxx.mdl \
  > nohup.txt 2>&1 & echo $! > pidfile.txt

# use this command to kill process:
# kill -9 `cat pidfile.txt`
