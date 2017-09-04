model_id=30
model_id_1=29
model_id_2=34
model_id_3=36
model_id_4=32
model_id_5=37
eargs=""

for num in 40 41 42 43 44 45 46 48 50 51 52 53 54 55; do
    eval_path="out/basic/${model_id}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done

for num in 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59; do
    eval_path="out/basic/${model_id_1}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done

for num in 40 41 42 43 44 45 46 47 48 49 50 51 52 53; do
    eval_path="out/basic/${model_id_2}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done

for num in 40 41 42 43 44 45 46 47 48 49; do
    eval_path="out/basic/${model_id_3}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done

for num in 43 44 45 46 47; do
    eval_path="out/basic/${model_id_4}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done

for num in 40 43 44 45 46 47 48; do
    eval_path="out/basic/${model_id_5}/eval/test-0${num}000.pklz"
    eargs="$eargs $eval_path"
done
#for num in 41 42 43 46; do
#    eval_path="out/basic/${model_id_2}/eval/test-0${num}000.pklz"
#    eargs="$eargs $eval_path"
#done
python3 -m basic.ensemble --data_path newsqa/data_test.json --shared_path newsqa/shared_test.json -o new_results_1.json $eargs
python3 newsqa/evaluate.py newsqa/data_test.json new_results_1.json

