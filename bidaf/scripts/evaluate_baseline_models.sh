# Evaluate baseline
python3 -m basic_old.cli \
--run_id 12 \
--load_path out/basic/06/save/basic-40000 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Evaluate generated data model with unfiltered questions 
python3 -m basic_old.cli \
--run_id 11 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-42000 \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Evaluate generated filtered data model
python3 -m basic_old.cli \
--run_id 12 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/12/save/basic-49000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 14 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 18 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/18/save/basic-41000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 16 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/16/save/basic-41000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 16 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/16/save/basic-47000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 16 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/17/save/basic-47000 \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 17 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 17 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Evaluate original squad
python3 -m basic_old.cli \
--run_id 23 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa_gen \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100


python3 -m basic_old.cli \
--run_id 22 \
--shared_path out/basic/00/shared.json \
--load_path out/basic/00/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 2 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 25 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 26 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Semi supervised full context spans
# Ratio: supervised 3 to unsup 1
python3 -m basic_old.cli \
--run_id 27 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Semi supervised full context spans
# Ratio: unsupervised 3 to supervised 1
python3 -m basic_old.cli \
--run_id 26 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 29 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-44000 \
--k 10 \
--use_special_token False \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 30 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-42000 \
--k 10 \
--use_special_token False \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# 5 vs 1
python3 -m basic_old.cli \
--run_id 30 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 3 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# 5 vs 1 
python3 -m basic_old.cli \
--run_id 31 \
--shared_path out/basic/06/shared.json \
--k 10 \
--load_ema False --gpu_idx 1 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

python3 -m basic_old.cli \
--run_id 32 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--load_ema False --gpu_idx 1 \
--sup_unsup_ratio 7 \
--mode test --data_dir newsqa \
--len_opt --batch_size 15 --num_steps 40000 \
--eval_period 1000 --save_period 1000 \
--sent_size_th 2100 --para_size_th 2100

# Evaluate semisupervised models 
25: Way more supervised than not
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/25/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/25/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/25/answer/test-044000.json


26: 
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/26/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/26/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/26/answer/test-046000.json

27:
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/27/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/27/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/27/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/27/answer/test-044000.json


29:
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-044000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-045000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/29/answer/test-050000.json

30: 
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-044000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-045000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-046000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-048000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/30/answer/test-050000.json

31
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/31/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json new_results.json
# Evaluate finetuned on NEWSQA: 
# Test set
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/09/answer/test-053000.json
# Dev set
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/09/answer/dev-053000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/06/answer/test-040000.json

# Evaluate RAW model on NewsQA:
# Test set
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/09/answer/test-040000.json

# Dev set
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/09/answer/dev-040000.json

python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/12/answer/dev-047000.json

# Evaluate SQUAD
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/12/answer/dev-047000.json

# Evaluate NewsQA
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/12/answer/dev-047000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/00/answer/dev-040000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/12/answer/dev-051000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/11/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/14/answer/test-041000.json


python3 newsqa/evaluate.py newsqa/data_test.json out/basic/11/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-041000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/12/answer/test-040000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/17/answer/test-043000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/17/answer/test-042000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/16/answer/test-051000.json
# New model
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/09/answer/test-040000.json
python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/22/answer/dev-040000.json

ls out/basic/27/answer

python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/23/answer/dev-040000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/16/answer/test-051000.json
python3 newsqa/evaluate.py newsqa/data_test.json out/basic/13/answer/test-041000.json

python3 newsqa/evaluate.py newsqa/data_test.json out/basic/13/answer/test-041000.json

python3 newsqa/evaluate.py newsqa/data_dev.json out/basic/18/answer/dev-041000.json
