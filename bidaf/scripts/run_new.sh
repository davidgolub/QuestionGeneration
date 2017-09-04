model_id=$1
save_path="${model_id}_all.json"
eargs=""

eval_paths="out/basic/${model_id}/eval/test-*"
count=0
max=5
for eval_path in $eval_paths;
do
    ((count++))
    if (("$count" < "$max"))
    then
        eargs="$eargs $eval_path"
    fi
done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o $save_path $eargs
python3 newsqa/evaluate.py newsqa/data_test.json $save_path