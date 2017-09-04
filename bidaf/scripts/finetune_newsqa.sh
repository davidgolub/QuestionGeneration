#!/bin/bash

# Train new model: with logits on top 
python3 -m basic_old.cli \
--run_id 8 \
--load_path out/basic/06/save/basic-40000 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode train --data_dir data/squad \
--len_opt --batch_size 30 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300

tensorboard --logdir=out/basic/06/log

# Train new model:
python3 -m basic_old.cli \
--run_id 9 \
--load_path out/basic/33/save \
--shared_path out/basic/33/shared.json \
--k 10 \
--num_sents_th 30 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir data/squad \
--len_opt --batch_size 30 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300

# Test new model
python3 -m basic_old.cli \
--run_id 9 \
--k 10 \
--load_path save/33/save \
--shared_path save/33/shared.json \
--load_ema False --gpu_idx 1 \
--dump_answer True \
--data_dir data/squad \
--len_opt --batch_size 10 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 900 --para_size_th 900
--num_sents_th 20

# TRAIN
python3 -m basic_old.cli \
--run_id 6 \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir data/squad \
--len_opt --batch_size 30 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300

tensorboard --logdir=out/basic/06/log

# TEST
python3 -m basic_old.cli \
--run_id 8 \
--shared_path out/basic/06/shared.json \
--k 3 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir data/squad \
--len_opt --batch_size 10 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# TRAIN
python3 -m basic_old.cli \
--run_id 7 \
--load_path out/basic/06/save/basic-40000 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode train --data_dir data/squad \
--len_opt --batch_size 30 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300


# TEST SQUAD->NEWSQA
python3 -m basic_old.cli \
--run_id 9 \
--load_path out/basic/06/save/basic-40000 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2000 --para_size_th 2000


# TEST SQUAD->NEWSQA TRAINED
python3 -m basic_old.cli \
--run_id 9 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2000 --para_size_th 2000


# TRAIN SQUAD->NEWSQA
python3 -m basic_old.cli \
--run_id 9 \
--shared_path out/basic/06/shared.json \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# TEST NEWSQA->SQUAD
python3 -m basic_old.cli \
--run_id 10 \
--load_path out/basic/00/save/basic-32000 \
--shared_path out/basic/00/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir data/squad \
--len_opt --batch_size 24 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300


# TRAIN NEWSQA->SQUAD
python3 -m basic_old.cli \
--run_id 10 \
--load_path out/basic/00/save/basic-32000 \
--shared_path out/basic/00/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode train --data_dir data/squad \
--len_opt --batch_size 24 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300

# Train SQUAD->NEWSQA
python3 -m basic_old.cli \
--run_id 11 \
--shared_path out/basic/06/shared.json \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800
#--load_path out/basic/06/save/basic-40000 \
#--shared_path out/basic/06/shared.json \

# Test new model
python3 -m basic_old.cli \
--run_id 12 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen_filtered \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Train model on filtered dataset 
python3 -m basic_old.cli \
--run_id 12 \
--shared_path out/basic/06/shared.json \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800
#--load_path out/basic/06/save/basic-40000 \

# Test model on filtered dataset v2 (only CNN and short Q)
python3 -m basic_old.cli \
--run_id 13 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_gen_filtered_v2 \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Unsupervised without verb filtering
python3 -m basic_old.cli \
--run_id 13 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_gen_filtered_unsupervised \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Unsupervised with verb filtering
python3 -m basic_old.cli \
--run_id 14 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_unsupervised_verb_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

python3 -m basic_old.cli --run_id 14 --shared_path out/basic/06/shared.json --load_path out/basic/14/save/basic-41000 --k 10 --load_ema False --gpu_idx 1 --mode test --data_dir newsqa_gen --len_opt --batch_size 15 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 2100 --para_size_th 2100

# Supervised verb filtering better
python3 -m basic_old.cli \
--run_id 15 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered_v2 \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Smaller stuff for question generation
python3 -m basic_old.cli \
--run_id 16 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/16/save/basic-52000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen_filtered_small \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

python3 -m basic_old.cli \
--run_id 17 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_gen_filtered_small_v2 \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

python3 -m basic_old.cli \
--run_id 25 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/18/save/basic-41000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen_filtered_unsupervised_verb_filtered_truncated \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 18 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/18/save/basic-41000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen_filtered_unsupervised_verb_filtered_truncated \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Semi supervised: 20k squad and 5k other shit
python3 -m basic_old.cli \
--run_id 19 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode train --data_dir 'newsqa_gen_filtered_unsupervised_verb_filtered_truncated' \
--len_opt --batch_size 24 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Semisupervised stuff
# Ratio: 5 unsup to 1 squad, truncated context
python3 -m basic_old.cli \
--run_id 26 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered_truncated \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Ratio: 1 unsup to 3 squad, truncated context
python3 -m basic_old.cli \
--run_id 25 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered_truncated \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Ratio: 1 unsup to 3 squad, untruncated context q
python3 -m basic_old.cli \
--run_id 27 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Ratio: 3 unsup to 1 squad, full context
python3 -m basic_old.cli \
--run_id 28 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Ratio: 1 unsup to 5 squad, full context
python3 -m basic_old.cli \
--run_id 29 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Ratio: 1 semi-sup to 5 squad, full context
python3 -m basic_old.cli \
--run_id 30 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered \
--len_opt --batch_size 20 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Run with special token:
python3 -m basic_old.cli \
--run_id 31 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 0 \
--mode train --data_dir newsqa_gen_filtered_unsupervised_verb_filtered \
--len_opt --batch_size 24 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# --load_path out/basic/06/save/basic-40000 \

# Run 11: Unsupervised QA

# Baseline model
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/09/answer/dev-053000.json

# Unsupervised QA
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/13/answer/dev-047000.json

# Supervised QA 
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/12/answer/test-049000.json
accuracy=0.1955, f1=0.3465, loss=8.7264
# Evaluate models
python3.5 newsqa/evaluate.py newsqa/data_test.json out/basic/00/answer/test-050000.json
python3.5 newsqa/evaluate.py newsqa/data_dev.json out/basic/00/answer/dev-041000.json
python3.5 newsqa/evaluate.py newsqa/data_dev.json out/basic/00/answer/dev-041000.json

# Submit
python3.5 newsqa/evaluate.py newsqa/data_dev.json out/basic/17/answer/dev-041000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/09/answer/dev-053000.json


# Run 2
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/16/answer/test-044000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/16/answer/test-049000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/18/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/05/answer/dev-032999.json

# Original squad
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/00/answer/test-031005.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/00/answer/dev-040000.json

#python3 newsqa/evaluate.py out/basic/12/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/18/answer/dev-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-049000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-054000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/14/answer/dev-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/14/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/16/answer/dev-052000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/25/answer/dev-041000.json
# Eval on dev set
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-049000.json


python3 newsqa/evaluate.py newsqa/data_test.json out/basic/09/answer/test-031999.json

python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/09/answer/dev-053000.json

# Test semisupervised stuff
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/26/answer/test-041000.json


# Unsupervised ans = 2, k = 0
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/17/answer/test-041000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/26/answer/test-042000.json
