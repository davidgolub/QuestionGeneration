#!/bin/bash

model_id=06
eargs=""
for num in 40 41 42 43 44; do
    eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results.json $eargs

