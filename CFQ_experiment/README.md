# Learning Algebraic Recombination for Compositional Generalization - CFQ

## Requirements

Our code is officially supported by Python 3.7. The main dependency is `pytorch`.
You could install all requirements by the following command:

```setup
pip install -r requirements.txt
```

## Data Files

`data/mcd1/train & dev & test: `CFQ dataset under MCD1 split

`data/mcd1/encode_tokens.txt: `CFQ encode tokens

`data/mcd1/decode_tokens.txt: `CFQ decode tokens

`data/mcd1/enct2dect: `Extracted candidate alignments

Due to the data files are too large, we only provide MCD1-split here, for MCD2 and MCD3 please see https://github.com/google-research/google-research/tree/master/cfq.

## Training

To train the model on CFQ datasets, please modify the configurations in `train.sh` script and run:

```
sh train.sh
```

The training log and model weights will be stored in the path `checkpoint/logs/` and `checkpoint/models/` respectively.

To achieve the best performance, we recommend to use the curriculum learning method mentioned in the paper: First set `lesson=11` in `train.sh` to train the model, then set `pretrained-model` as the path of the trained model weights and set `lesson=30` to train further on the complete training set.

## Evaluation

To evaluate the model on test set, please modify the configurations in `eval.sh` script and run:

```
sh eval.sh
```


## Pre-trained Models

You can find trained model weights under the `checkpoint/models/` folder.



