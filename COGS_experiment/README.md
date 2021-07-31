# Learning Algebraic Recombination for Compositional Generalization - COGS

This repository is the extention implementation on COGS, based on the basic idea in our paper "Learning Algebraic Recombination for Compositional Generalization". 


## Requirements

Our code is officially supported by Python 3.7. The main dependency is `pytorch`.
You could install all requirements by the following command:

```setup
pip install -r requirements.txt
```

## Data Files

`cogs_data: `COGS dataset containing train, dev and test.

`preprocess: `Contains encode tokens, decode tokens and phrase table.

## Training

To train our model on COGS datasets, you could use this command:

```train
python main.py --mode train --checkpoint <model_dir> --task cogs
```

📋 Note that `<model_dir>` specifies the store folder of model checkpoints.

The corresponding log and model weights will be stored in the path `checkpoint/logs/` and `checkpoint/models/` respectively

## Evaluation

The accuracy on gen set will be printed as the last line in log after training.



