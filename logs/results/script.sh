# Evaluate out of domain baseline
echo "out of domain baseline" 
python3 newsqa/evaluate.py newsqa/data_test.json answer_out_of_domain_baseline.json

# Evaluate single model results (for steps 2k-10k)

# Evaluate single model results + baseline (for steps 2k-10k)

# Evaluate single-model result (44.5 F1)
echo "A_(gen + Ner)"
python3 newsqa/evaluate.py newsqa/data_test.json single_model.json 

# Evaluate two-model result (45.6 F1)
echo "Double model result-2 A_(gen + ner) + A_ner"
python3 newsqa/evaluate.py newsqa/data_test.json double_model.json

# Evaluate AOracle + Context
echo "Answer oracle with context for question generation, single model"
python3 newsqa/evaluate.py newsqa/data_test.json context_aoracle.json

echo "Single BiDAF model finetuned on NewsQA 4k steps"
python3 newsqa/evaluate.py newsqa/data_test.json "single_model_results_44.json"

echo "Single BiDaf finetuned on NewsQA 4k steps ensembled w. baseline results"
python3 newsqa/evaluate.py newsqa/data_test.json "single_model_result_run_44_with_baseline.json"

# Evaluate single model result of BiDAF finetuned on NewsQA
echo "Single BiDAF model finetuned on NewsQA results"
for num in 42 43 44 45 46 47 48 49; do
    python3 newsqa/evaluate.py newsqa/data_test.json "single_model_results_${num}.json"
done

echo "Single BiDAF model finetuned on NewsQA ensembled w. baseline results"
# Evaluate single model ensembled with baseline result of BiDAF finetuned on NewsQA
for num in 42 43 44 45 46 47 48 49; do
    python3 newsqa/evaluate.py newsqa/data_test.json "single_model_result_run_${num}_with_baseline.json"
done
