#!/bin/bash
/dfs/scratch0/golubd/reading_comprehension

# Installing everything
sudo pip3 install -r requirements.txt
chmod +x download.sh; ./download.sh
python3 -m squad.prepro

# Install spacy for NER tagging
python3 -m spacy download en

sudo python3.5 -m pip install -r requirements.txt

# Tensorflow stuff
tensorboard --logdir=/home/golubd/reading_comprehension/out/basic/03/log
tensorboard --logdir=/home/ubuntu/reading_comprehension/out/basic/00/log
tensorboard --logdir=/dfs/scratch0/reading_comprehension/out/basic/04/log

# NewsQA stuff
python3 -m newsqa.prepro

# Train operation: newsqa
python3.5 -m basic.cli --mode train --noload --debug # First test everything works
python3.5 -m basic.cli --mode train --noload --len_opt --cluster --batch_size 28 --num_steps 40000 # Run fast training

# Train operation: newsqa 1
python3.5 -m basic.cli --run_id 0 --noload --mode train --len_opt --cluster --batch_size 24 --num_steps 40000 --sent_size_th 800 --para_size_th 800
python3.5 -m basic.cli --run_id 0 --gpu_idx 1 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train

# Test loading operations
python3.5 -m basic.cli --run_id 1 --mode train --data_dir newsqa --len_opt --batch_size 28 --num_steps 40000 --save_period 1 --eval_period 0

# Train operation: squad
python3.5 -m basic.cli --run_id 1 --mode train --data_dir newsqa --len_opt --batch_size 28 --num_steps 40000 --save_period 0 --eval_period 0# Run fast training
python3.5 -m basic.cli --run_id 1 --reinforce_train --mode train --data_dir data/squad --len_opt --batch_size 28 --num_steps 40000 --save_period 2 --eval_period 2 # Run fast training

python3.5 -m basic.cli --run_id 1 --mode train --data_dir newsqa --len_opt --batch_size 28 --num_steps 40000 --save_period 2 --eval_period 2 # Run fast training
python3.5 -m basic.cli --run_id 0 --mode train --len_opt --cluster --batch_size 24 --num_steps 20000 --sent_size_th 800 --para_size_th 800 --save_period 10 --eval_period 10
python3.5 -m basic.cli --run_id 0 --mode train --len_opt --cluster --batch_size 24 --num_steps 20000 --sent_size_th 800 --para_size_th 800 --save_period 2 --eval_period 2

# Test operation: squad
python3.5 -m basic.cli --run_id 1 --data_dir newsqa --len_opt --cluster --batch_size 5 --sent_size_th 600 --para_size_th 600
python3.5 -m basic.cli --run_id 1 --data_dir data/squad --len_opt --cluster --batch_size 28 --sent_size_th 600 --para_size_th 600

# Test operation: squad
python3.5 -m basic.cli --run_id 1 --mode train --data_dir newsqa --len_opt --batch_size 28 --num_steps 40000 --eval_period 50 --save_period 50 --noload 

# Test operation:
python3.5 -m basic.cli --len_opt --cluster --batch_size 10 --sent_size_th 2100 --para_size_th 2100
python3.5 -m basic.cli --run_id 0 --reinforce_train --len_opt --cluster --batch_size 24 --sent_size_th 800 --para_size_th 800

# Evaluate stuff
python3.5 newsqa/evaluate.py newsqaqa/data_test.json out/basic/00/answer/test-####.json
python3.5 newsqa/evaluate.py newsqa/data_test.json out/basic/00/answer/test-016000.json

python3.5 newsqa/evaluate.py newsqa/data_test.json out/basic/00/answer/test-040000.json
python3.5 newsqa/evaluate.py newsqa/data_test.json out/basic/00/answer/test-045000.json

python3.5 newsqa/evaluate.py newsqa/data_dev.json out/basic/00/answer/dev-041000.json
# Gets {"exact_match": 41.45532579008974, "f1": 55.19630616316198}

python3.5 squad/evaluate-v1.1.py squad/dev-v1.1.json out/basic/00/answer/dev-041000.json

# Evaluate using an ensemble:python3.5 squad/evaluate-v1.1.py squad/dev-v1.1.json out/basic/00/answer/dev-041000.json
python3.5 squad/evaluate-v1.1.py squad/dev-v1.1.json out/basic/09/answer/test-018000.json

basic/run_ensemble.sh newsqa/test.csv ensemble.json 


# Test loading:
du -a /afs/cs.stanford.edu/u/golubd/ | sort -n -r | head -n 5
### NEWSQA
#Newsqa old
python3 -m basic.cli --run_id 0 --load_ema False --gpu_idx 1 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train
python3 -m basic.cli --run_id 0 --load_ema False --gpu_idx 1 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train

python3 -m basic.cli --run_id 0 --load_ema False --gpu_idx 1 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train

#Newsqa another run
python3 -m basic.cli --run_id 0 --load_ema False --gpu_idx 0 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train 
python3 -m basic.cli --run_id 3 --load_ema False --gpu_idx 3 --mode train --data_dir newsqa --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 800 --para_size_th 800 --reinforce_weight 0.00 --reinforce_train --noload

#Newsqa another run
#SQUAD
python3 -m basic.cli --run_id 1 --gpu_idx 2 --mode train --data_dir data/squad --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 300 --para_size_th 300 --reinforce_weight 0.00 --reinforce_train --load_ema False

python3 -m basic.cli --run_id 2 --gpu_idx 3 --mode train --data_dir data/squad --len_opt --batch_size 24 --num_steps 40000 --eval_period 1000 --save_period 1000 --sent_size_th 300 --para_size_th 300 --reinforce_weight 0.00 --reinforce_train --noload

# Tests
python -m tests.load_dev_results