#!/bin/bash

# First download all of the embeddings
# Requires python 2.7
python2.7 -m pretrained_models.scripts.download_glove_embeddings
python2.7 -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/squad/vocab.txt' \
--save_embeddings_path 'datasets/squad/word_embeddings.npy'

python2.7 -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/squad_iob/vocab.txt' \
--save_embeddings_path 'datasets/squad_iob/word_embeddings.npy'

python2.7 -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/newsqa_unsupervised/vocab.txt' \
--save_embeddings_path 'datasets/newsqa_unsupervised/word_embeddings.npy'

python2.7 -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/newsqa_unsupervised_large/vocab.txt' \
--save_embeddings_path 'datasets/newsqa_unsupervised_large/word_embeddings.npy'

# Run tests to make sure model makes on a simple example
python3 -m tests.language_model_trainer_test

# Run question generator: with entire vocab
python3 -m tests.squad_trainer_test # Entire context

# Generate NewsQA ans + paragraph data
cd bidaf 
python3 -m tests.create_generation_dataset_unsupervised

# Copy vocabulary.txt file to proper location, and generate word embeddings
cp datasets/squad/vocab.txt datasets/newsqa_unsupervised/vocab.txt

# Generate word_embeddings.npy file that corresponds to vocab.txt file
python2.7 -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/newsqa_unsupervised/vocab.txt' \
--save_embeddings_path 'datasets/newsqa_unsupervised/word_embeddings.npy'

# Now run predictions on NewsQA ans + paragraph data.
python3 -m tests.newsqa_predictor_test_unsup # 250K questions

# Now create the synthetic NewsQA dataset
cd ../
cd bidaf
python3 -m tests.create_bidaf_dataset

# Create old dataset, for reference
python3 -m tests.create_bidaf_old_dataset 

# Link shared files
ln newsqa/data_dev.json newsqa_unsupervised/data_dev.json
ln newsqa/data_test.json newsqa_unsupervised/data_test.json
ln newsqa/shared_dev.json newsqa_unsupervised/shared_dev.json
ln newsqa/shared_test.json newsqa_unsupervised/shared_test.json
ln newsqa/shared_train.json newsqa_unsupervised/shared_train.json

ln newsqa/data_dev.json newsqa_unsupervised_old/data_dev.json
ln newsqa/data_test.json newsqa_unsupervised_old/data_test.json
ln newsqa/shared_dev.json newsqa_unsupervised_old/shared_dev.json
ln newsqa/shared_test.json newsqa_unsupervised_old/shared_test.json
ln newsqa/shared_train.json newsqa_unsupervised_old/shared_train.json

ln newsqa/data_dev.json newsqa_gen_filtered_unsupervised_verb_filtered/data_dev.json
ln newsqa/data_test.json newsqa_gen_filtered_unsupervised_verb_filtered/data_test.json
ln newsqa/shared_dev.json newsqa_gen_filtered_unsupervised_verb_filtered/shared_dev.json
ln newsqa/shared_test.json newsqa_gen_filtered_unsupervised_verb_filtered/shared_test.json
ln newsqa/shared_train.json newsqa_gen_filtered_unsupervised_verb_filtered/shared_train.json

# Now run training with squad and synthetic dataset
python3 -m basic.cli \
--run_id 17 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 5 \
--sup_unsup_ratio 3 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_unsupervised \
--len_opt --batch_size 24 --num_steps 14000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Now run training with squad and old dataset
python3 -m basic.cli \
--run_id 21 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--sup_unsup_ratio 5 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_unsupervised_old \
--len_opt --batch_size 24 --num_steps 14000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Now run training with squad and old dataset, verb filtered
python3 -m basic.cli \
--run_id 22 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--sup_unsup_ratio 5 \
--load_ema False --gpu_idx 3 \
--mode train --data_dir newsqa_unsupervised_old_verb_filtered \
--len_opt --batch_size 24 --num_steps 14000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800

# Now run training with squad and old dataset, verb filtered
python3 -m basic.cli \
--run_id 22 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--sup_unsup_ratio 5 \
--load_ema False --gpu_idx 2 \
--mode train --data_dir newsqa_unsupervised_old \
--len_opt --batch_size 20 --num_steps 14000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800


python3 -m basic.cli \
--run_id 18 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 5 \
--sup_unsup_ratio 2 \
--load_ema False --gpu_idx 2 \
--mode train --data_dir newsqa_unsupervised \
--len_opt --batch_size 24 --num_steps 20000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 800 --para_size_th 800


# Once that's finished run evalations on the saved models
# Creates pklz files that can be used for final eval.
for i in 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59;
do 
    for j in 20 21;
    do
        python3 -m basic.cli \
        --run_id $j \
        --shared_path out/basic/06/shared.json \
        --load_path "out/basic/$j/save/basic-"$i"000" \
        --k 10 \
        --use_special_token False \
        --load_ema False --gpu_idx 3 \
        --mode test --data_dir newsqa \
        --len_opt --batch_size 15 --num_steps 40000 \
        --eval_period 1000 --save_period 1000 \
        --sent_size_th 2100 --para_size_th 2100
    done
done

# Baseline eval
python3 -m basic.cli \
    --run_id 14 \
    --shared_path out/basic/06/shared.json \
    --load_path "out/basic/06/save/basic-40000" \
    --k 10 \
    --use_special_token False \
    --load_ema False --gpu_idx 1 \
    --mode test --data_dir newsqa \
    --len_opt --batch_size 15 --num_steps 20000 \
    --eval_period 1000 --save_period 1000 \
    --sent_size_th 2100 --para_size_th 2100

model_id=19
eargs=""

for num in 45; do
    eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_30.json
python3 -m basic_old.cli \
    --run_id 06 \
    --shared_path out/basic/06/shared.json \
    --load_path "out/basic/06/save/basic-40000" \
    --k 10 \
    --use_special_token False \
    --load_ema False --gpu_idx 2 \
    --mode test --data_dir newsqa \
    --len_opt --batch_size 15 --num_steps 40000 \
    --eval_period 1000 --save_period 1000 \
    --sent_size_th 2100 --para_size_th 2100


export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL


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

# Joint evaluation
eargs=""
for num in 42 42 43 44 45 46 47 48 49 51; do
    for model_id in 20 21; do
        eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
        eargs="$eargs $eval_path"
    done
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json double_model.json


# Joint evaluation with baseline
eargs=""
for num in 41 42 43 44 45 46 47 48 49 51; do
    for model_id in 20; do
        eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
        eargs="$eargs $eval_path"
    done
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs

for num in 42 43 44 45 46 47 48 49; do
    for model_id in 21; do
        eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
        eargs="$eargs $eval_path"
    done
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_30.json


