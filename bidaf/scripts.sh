# Now do evaluations on the pklz files with predictions
model_id=14
eargs=""

for num in 40; do
    eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
#for num in 41 42 43 46; do
#    eval_path="out/basic/${model_id_2}/eval/test-0${num}000.pklz"
#    eargs="$eargs $eval_path"
#done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_30.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_30.json
