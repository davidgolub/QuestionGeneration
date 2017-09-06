Code for [Two-Stage Synthesis Networks for Transfer Learning in Machine Comprehension](https://arxiv.org/abs/1706.09789).

We provide our components, which include a clean PyTorch implementation of [Latent Predictor Networks](https://arxiv.org/abs/1603.06744), an NER answer chunker, and a procedure to finetune a [BIDAF](https://arxiv.org/abs/1611.01603) model on a combination of synthetic and real question, answer pairs.


We provide the pre-generated synthetic question answer pairs and a pretrained SQuAD BiDAF model that can be trained on NewsQA data.

The question-generation network takes in a passage, extracts answer spans from the passage, and for each answer span, generates a question. In our work, we use the question generator network to finetune a Reading Comprehension Model trained on [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/) to answer questions on [NewsQA](https://datasets.maluuba.com/NewsQA). 

Finally, we also provide several logs from our experiments for single-model, two-model results, and gold answer finetuning under logs/results.

Prerequisites
-------------
- Git LFS

- Python 3.5+
- [Pytorch](https://www.pytorch.org/)
- [Tensorflow](https://tensorflow.org/) version 0.12
- NumPy
- NLTK
- CUDA
- tqdmc

- Python 2.7
- tqdm
- unidecode
- textblob

Quickstart
----------
* To setup the NewsQA dataset, please download and preprocess the NewsQA dataset from [Maluuba's NewsQA][maluuba] repository. Afterwards, please place the train.csv, test.csv and dev.csv files into bidaf/newsqa. 
Then, to setup the NewsQA data, please run these commands:
```
cd bidaf
./download.sh
python3 -m newsqa.prepro
```

To install necessary dependencies, please run 
```
cd ../bidaf
./install.sh
```

To get remaining datasets, please run
```
git lfs pull origin master
```

For a preliminary example of how to extract answers (currently NER), generate questions, and then finetune a BIDAF model on the data, see 
```
./scripts.sh. 
```

For an example of how to finetune a BiDAF model trained on SQuAD on NewsQA using our old logs, please follow the instructions in 
```
./scripts.sh
```
and from the bidaf directory run 
```
# Now run training with squad and old dataset
python3 -m basic.cli \
--run_id 22 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--sup_unsup_ratio 5 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_unsupervised_old \
--len_opt --batch_size 24 --num_steps 14000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

for i in 41 42 43 44 45 46 47 48 49 50 51 52 53 54;
do
    python3 -m basic.cli \
    --run_id 22 \
    --shared_path out/basic/06/shared.json \
    --load_path "out/basic/22/save/basic-"$i"000" \
    --k 10 \
    --use_special_token False \
    --load_ema False --gpu_idx 3 \
    --mode test --data_dir newsqa \
    --len_opt --batch_size 15 --num_steps 40000 \
    --eval_period 1000 --save_period 1000 \
    --sent_size_th 2100 --para_size_th 2100
done

eargs=""
model_id=22
for num in 41 42 43 44 45 46 47 48 49 51; do
    eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_30.json
```

(running this command on my machine, this gives approximately ~30.5 EM and 44.5 F1 performance).


To run several of our logs, please execute:
```
cd logs/results
bash script.sh
```

For an end-to-end example of how to train your own question generator network, run
```
$ python3 -m tests.language_model_trainer_test 
```

For an end-to-end example of how to train your own answer chunking model, run
```
$ python3 -m tests.iob_trainer_test
```

A pre-trained BIDAF SQuAD model can be found at bidaf/out/basic/06/save/*
Synthetic question, answer pair datasets can be found at bidaf/newsqa_unsupervised_old (better performance) and bidaf/newsqa_unsupervised_old_verb_filtered (worse performance) 

**Question Generation**
Please note, to use a question generation network on SQuAD to generate questions on NewsQA, you must first create an inputs.txt file which corresponds to the paragraphs in CNN/Daily Mail. For legal reasons we can't provide it as part of the repository. To create them, please run
```
cd bidaf && python3 -m tests.create_generation_dataset_unsupervised
cd ../
cp datasets/newsqa_unsupervised/train/inputs.txt datasets/{NEWSQA_DATASET_OF_YOUR_CHOICE}/train/inputs.txt
```

Code Organization
-----
**datasets**
Contains sample datasets used to train the model. C.f. datasets/question_generation. Each dataset needs to have a vocab.txt file, inputs.txt, outputs.txt etc.  

**data_loaders**
Contains code to load a dataset from a directory into memory, and generate batches of examples to train/validate a model.

**models**
Contains core code for the question generator network (language_model.py),  IOB tagging model (iob/iob_model.py), and trainer (language_trainer.py)

**tests**
Contains unit tests for loading, training, predicting the network, and other components of the stack.
newsqa_predictor/* contains tests for predicting on newsqa.
squad_predictor/* contains test for predicting on squad.

**helpers**
Contains various utilities for loading, saving, things that make pytorch easier to work with.

**dnn_units**
Contains the core LSTM units for encoding/decoding.

**trainers**
Contains a trainer for training the answer chunker model.

**bidaf**
Contains the necessary code for training a reading comprehension model. This code is heavily based on the [Bi-directional Attention Flow for Machine Comprehension repository][bidaf] (thanks to authors for releasing their code!)

[maluuba]: https://github.com/Maluuba/newsqa
[cnn_stories]: http://cs.nyu.edu/~kcho/DMQA/
[bidaf]: https://github.com/allenai/bi-att-flow
