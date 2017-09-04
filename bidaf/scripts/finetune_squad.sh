python3 -m basic_old.cli \ 
--run_id 29 \ 
--use_special_token False \ 
--sup_unsup_ratio 5 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--baseline_dir newsqa \
--load_ema False --gpu_idx 0 \
--num_gpus 0 \
--mode train \
--data_dir squad_train_unsupervised_verb_filter \
--len_opt --batch_size 30 \
--num_steps 40000 \ 
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300 

python3 -m basic_old.cli \ 
--run_id 30 \ 
--use_special_token False \ 
--sup_unsup_ratio 3 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--baseline_dir newsqa \
--load_ema False --gpu_idx 0 \
--num_gpus 0 \
--mode train \
--data_dir squad_train_unsupervised_verb_filter \
--len_opt --batch_size 30 \
--num_steps 40000 \ 
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300

python3 -m basic_old.cli \ 
--run_id 31 \ 
--use_special_token False \ 
--sup_unsup_ratio 5 \
--shared_path out/basic/06/shared.json \
--load_path out/basic/06/save/basic-40000 \
--k 10 \
--baseline_dir newsqa \
--load_ema False --gpu_idx 0 \
--num_gpus 0 \
--mode train \
--data_dir squad_train_unsupervised_verb_filter_iob \
--len_opt --batch_size 30 \
--num_steps 40000 \ 
--eval_period 1000 --save_period 1000 \
--sent_size_th 300 --para_size_th 300  
