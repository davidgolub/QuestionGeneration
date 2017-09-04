#!/bin/bash
python -m pretrained_models.scripts.download_glove_embeddings
python -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/squad/vocab.txt' \
--save_embeddings_path 'datasets/squad/word_embeddings.npy'

python -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/newsqa_unsupervised/vocab.txt' \
--save_embeddings_path 'datasets/newsqa_unsupervised/word_embeddings.npy'

python -m pretrained_models.scripts.transfer_glove_embeddings --path 'pretrained_models/word_embeddings/glove/glove.840B.300d.txt' \
--save_word_path 'datasets/newsqa_unsupervised_large/vocab.txt' \
--save_embeddings_path 'datasets/newsqa_unsupervised_large/word_embeddings.npy'