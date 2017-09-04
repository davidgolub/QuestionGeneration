# Finetuning BiDAF with SynthNets: 
 
- This repository implements finetuning a [Bi-directional Attention Flow for Machine Comprehension] (Seo et al., 2016) model trained on a source collection of documents to answer questions on a target set of documents using [Two-stage SynthNets]. It assumes a SynthNet already generated question, answer tuples over the desired set.

## 0. Requirements
#### General
- Python (verified on 3.5.2. Issues have been reported with Python 2!)
- unzip, wget (for running `download.sh` only)

#### Python Packages
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Downloading Data
Run:
```
git lfs pull
```

## Scripts
All commands used to train, test models are stored under [scripts].

Each command inside each script file should be run from the root directory of the repository.

## 2. Training
To finetune a pretrained [SQUAD] BIDAF model on [NewsQA], see the scripts at [scripts/finetune_newsqa].

To finetune a pretrained [NewsQA] model on [SQUAD], see the scripts at [scripts/finetune_squad].

## 3. Test
To evaluate single models, see the scripts at scripts/evaluate_*.sh.

To evaluate intra-run averaged models, ensembles, etc. see the scripts at scripts/*_evaluation.sh

[Two-stage SynthNets]: https://arxiv.org/TODO
[Bi-directional Attention Flow for Machine Comprehension]: https://github.com/allenai/bi-att-flow
[scripts]: https://github.com/davidgolub/ReadingComprehension/tree/master/scripts
[scripts/finetune_newsqa]: https://github.com/davidgolub/ReadingComprehension/tree/master/scripts/finetune_newsqa.sh
[scripts/finetune_squad]: https://github.com/davidgolub/ReadingComprehension/tree/master/scripts/finetune_squad.sh
[code]: https://github.com/allenai/bi-att-flow
[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[SQUAD]: http://stanford-qa.com
[NEWSQA]: https://datasets.maluuba.com/NewsQA
[paper]: https://arxiv.org/abs/1611.01603
[davidgolub]: https://davidgolub.github.io
[davidgolub-github]: https://github.com/davidgolub
