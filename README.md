# Sentiment (Emotion) Chatbot
This is a simple single-turn chatbot for Korean emotional dialogue. We implemented and trained GPT-2 base model **from scratch**. Furthermore, we used a technique that predicts the emotion for the user input and utilized it as a prompt.

We used the [Emotional dialogue corpus](https://aihub.or.kr/aidata/7978) dataset provided by AI Hub. Data files are available in `./data/raw`. Our code will pre-process and store them on `./data/processed`.

\* This repository contains the source code of the Team 15 of Artificial Intelligence Project (SWE3032-41, 2021Fall) at SKKU.

## How to use
At root dir,
- Train  
`python -m src.main`

- Test (test with your input)  
`python -m src.test --weight-path [WEIGHT PATH]`  


**CAUTION: There are occasional errors in input recognition, so proper output can be obtained only by entering input without typos at a time.**

You can adjust the train setting by arguments. (See below)

## Arguments
- --seed: Random seed (default 42)
- --batch_size: Batch size (default 64)
- --n-epochs: Number of epochs to train (default 100)
- --learning rate: Learning rate (default 1e-4)
- --data-path: Path of data files (default data/)
- --base-tokenizer: Base tokenizer to use (default LTokenizer)
- --gen-policy: Generation policy (default greedy)
- --gen-ex-input: Sample input to check every epoch (default '나 요즘 너무 우울해')
- --logging-step: Logging step (default 150)
- --DEBUG / --NO-DEBUG: In debugging mode, unable Wandb, other things are exactly the same with no-debugging mode (default NO-DEBUG)

## Requirements
### Dependencies
Python 3.7+  
PyTorch 1.10.0  
   
- numpy==1.20.1
- pandas==1.2.4
- wandb==0.12.7
- soynlp==0.0.493
- nltk==3.6.1

### How to install
```
pip install -r requirements.txt
```

## File Structure
```
./src
├── main.py - main (execute)
├── test.py - test (execute with your own input)
├── model.py - contain all models
├── option.py - options (arguments, model configs, etc.)
├── train.py - train/eval/test
└── utils - utilities
    ├── dataset.py - dataset & dataloader & vocabulary
    ├── generate.py - generate function
    ├── lr_scheduler.py - custom learning scheduler
    ├── metric.py - metrics
    ├── preprocessing.py - dataset preprocessing & save
    ├── tokenizer.py - tokenizer
    └── utils.py - others (set seed, num_workers, etc.)
```

## Referneces
[1] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.(http://www.persagen.com/files/misc/radford2019language.pdf)

[2] Multi-turn chatbot project (3): GPT-2 chatbot with multi-turn generation settings. (Nov 28, 2020). (https://songstudio.info/tech/tech-35/)

[3] The Annotated GPT-2. (Feb 18, 2020). (https://amaarora.github.io/2020/02/18/annotatedGPT2.html)

[4] LSTM Text Classification Using Pytorch. (Jul 1. 2020). (https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0)