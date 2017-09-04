#!/bin/bash
python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 out/basic/06/answer/test-040000.json \
-model2 out/basic/00/answer/test-040000.json \
-name1 "BIDAF out-domain" \
-name2 "BIDAF in-domain" \
-output "BIDAF_results/outdomain_vs_indomain"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 out/basic/30/answer/test-044000.json \
-model2 out/basic/00/answer/test-040000.json \
-name1 "BIDAF Synthetic, k=5, fake a, fake q" \
-name2 "BIDAF on NewsQA" \
-output "BIDAF_results/k_5_single_vs_indomain"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 29_all.json \
-model2 out/basic/06/answer/test-040000.json \
-name1 "BIDAF Synthetic, k=3, intra ensemble" \
-name2 "BIDAF on SQUAD -> Newsqa" \
-output "BIDAF_results/synthetic_k_3_intra_out_domain"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 30_all.json \
-model2 out/basic/06/answer/test-040000.json \
-name1 "BIDAF Synthetic, k=5, intra ensemble" \
-name2 "BIDAF on SQUAD -> Newsqa" \
-output "BIDAF_results/synthetic_k_5_intra_out_domain"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 30_all.json \
-model2 26_all.json \
-name1 "BIDAF Synthetic, k=5, intra ensemble" \
-name2 "BIDAF Synthetic, k=0, intra ensemble" \
-output "BIDAF_results/synthetic_k_5_intra_k_0_intra"


python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 30_all.json \
-model2 29_all.json \
-name1 "BIDAF Synthetic, k=5, intra ensemble" \
-name2 "BIDAF Synthetic, k=3, intra ensemble" \
-output "BIDAF_results/synthetic_k_5_intra_k_3_intra"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model1 30_all.json \
-model2 out/basic/00/answer/test-040000.json \
-name1 "BIDAF Synthetic, k=5, fake a, fake q" \
-name2 "BIDAF on SQUAD -> Newsqa" \
-output "BIDAF_results/synthetic_k_5_intra_vs_indomain"

python3 -m visualization.compare_models_newsqa \
-dataset newsqa/data_test.json \
-model2 30_all.json \
-model1 12_all.json \
-name1 "BIDAF Synthetic, k=5, fake a, fake q" \
-name2 "BIDAF on SQUAD -> Newsqa" \
-output "BIDAF_results/synthetic_k_5_intra_vs_k_0_real_ans"

