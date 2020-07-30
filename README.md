# Additional Masked Language Model Pretrain for BERT

Pretrained language models such as BERT have shown to be superb on many NLP tasks. However, it is possible that your NLP task can be related to one specific domain (eg. finance, science and medicine) in which case you might consider taking additional MLM pretraining on a large corpus. (one related to the task's domain)  

This method is known as **Domain Adaptive Pretraining**, as suggest from [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) which has shown to be benenficial for various NLP tasks in terms of precision.  

In this repo, I have used two variants of Korean BERT  
- [Korean BERT](https://github.com/SKTBrain/KoBERT) from SKT-Brain, with 12-layer of Transformer block  
- Its distilled version, [DistilBERT](https://github.com/monologg/DistilKoBERT), with 3-layer of Transformer block

Then, I have  carried out additional MLM pretraining with Korean NLI dataset, released from   [KakaoBrain](https://www.kakaobrain.com/publication/124) where each row is of `sent1 \t sent2 \t label`  

## Process
### 1. Preprocess of NLI data
- Place your own data under `./data` directory  
- Depending on the format of your data (whether each row is of single sentence or multiple sentences) you might need to modify [preprocess.py](./preprocess.py)

### 2. Additional PreTrain
- run train.py
- `distill` argument is set to True by default, in which case a distilled Korean BERT is pretrained. You need to change it to False in order to pretrain 12-layered BERT

## MLM Pretraining
```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules
1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

## Acknowledgement
- [codertimo](https://github.com/codertimo/BERT-pytorch)