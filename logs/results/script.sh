# Evaluate out of domain baseline
echo "out of domain baseline" 
python3 newsqa/evaluate.py newsqa/data_test.json answer_out_of_domain_baseline.json

# Evaluate single-model result (44.5 F1)
echo "A_(gen + Ner)"
python3 newsqa/evaluate.py newsqa/data_test.json single_model.json 

# Evaluate two-model result (45.6 F1)
echo "Double model result-2 A_(gen + ner) + A_ner"
python3 newsqa/evaluate.py newsqa/data_test.json double_model.json

# Evaluate AOracle + Context
echo "Answer oracle with context for question generation, single model"
python3 newsqa/evaluate.py newsqa/data_test.json context_aoracle.json
