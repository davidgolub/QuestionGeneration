#!/bin/bash
model_id=$1
save_path="${model_id}_all.json"
eval_paths="out/basic/${model_id}/eval/test-*"
eargs=""
for eval_path in $eval_paths;
do
    eargs="$eargs $eval_path"
done

python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o $save_path $eargs
python3 newsqa/evaluate.py newsqa/data_test.json $save_path