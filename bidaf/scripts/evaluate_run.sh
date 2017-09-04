#!/bin/bash


for i in 42 43 44 45 46 47 48 49;
do 
    python3 -m basic_old.cli \
    --run_id 18 \
    --shared_path out/basic/06/shared.json \
    --load_path "out/basic/18/save/basic-"$i"000" \
    --k 10 \
    --use_special_token False \
    --load_ema False --gpu_idx 1 \
    --mode test --data_dir newsqa \
    --len_opt --batch_size 15 --num_steps 40000 \
    --eval_period 1000 --save_period 1000 \
    --sent_size_th 2100 --para_size_th 2100
done

#for i in 42 43 44 45 46 47 48 49;
#do 
#    python3 -m basic_old.cli \
#    --run_id 14 \
#    --shared_path out/basic/06/shared.json \
#    --load_path "out/basic/14/save/basic-"$i"000" \
#   --k 10 \
#    --use_special_token False \
#    --load_ema False --gpu_idx 3 \
#    --mode test --data_dir newsqa \
#    --len_opt --batch_size 15 --num_steps 40000 \
#    --eval_period 1000 --save_period 1000 \
#    --sent_size_th 2100 --para_size_th 2100
#done