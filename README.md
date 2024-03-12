AAAI24@LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack

LimeAttack's code:

## Requirements

- py3.10
- boto3==1.26.28
- botocore==1.29.28
- torch == 1.12.1+cu116
- tensorflow-gpu == 2.11.0（optional）
- tensorflow-hub == 0.12.0（optional）
- numpy == 1.23.2
- nltk == 3.7
- scipy == 1.9.1



## Datesets and Victim Model
There are  MR, SST-2 , AG, Yahoo and  SNLI, MNLI and MNLIm datasets. 
We adopt the pretrained models provided including BERT,CNN,LSTM. These data and models are adopted from  [HLBB](https://arxiv.org/abs/2012.14956) or [TextAttack](https://textattack.readthedocs.io/en/latest/). 


## Dependencies
glove.6B.200d.txt and  counter-fitted-vectors.txt  can be obtained from  [TextFooler](https://github.com/jind11/TextFooler)


## File Description
- LimeAttack_classification.py: Attack the victim model for text classification with LimeAttack.

## run
bash attack_mr.sh
