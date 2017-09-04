#!/bin/bash
# Once that's finished run evalations on the saved models
# Creates pklz files that can be used for final eval.
for i in 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59;
do 
    for j in 17 18 19;
    do
        python3 -m basic.cli \
        --run_id $j \
        --shared_path out/basic/06/shared.json \
        --load_path "out/basic/$j/save/basic-"$i"000" \
        --k 10 \
        --use_special_token False \
        --load_ema False --gpu_idx 3 \
        --mode test --data_dir newsqa \
        --len_opt --batch_size 10 --num_steps 40000 \
        --eval_period 1000 --save_period 1000 \
        --sent_size_th 2100 --para_size_th 2100
    done
done
for num in 40 41 42 43 44 45; do
    eval_path="out/basic/14/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_30.json
