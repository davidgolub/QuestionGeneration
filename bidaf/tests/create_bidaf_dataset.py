import json
import argparse
import os
import re
import sys
import time
import urllib
from urllib.parse import quote
from bs4 import BeautifulSoup
from urllib.request import urlopen
from helpers import utils
from collections import defaultdict 
from itertools import groupby 

def dedup(q):
  grouped_L = [[k, sum(1 for i in g)] for k,g in groupby(q)]
  deduped_q = list(map(lambda l: l[0], grouped_L))
  #if "?" not in deduped_q:
  #  print("Adding new question")
  #  deduped_q.append("?")
  return deduped_q 

def invalid_question(q):
  string_q = ' '.join(q)
  cnn_test = "CNN" in string_q
  unk_test = "<UNK>" in string_q
  q_test = "?" not in string_q
  small_q_test = len(dedup(q)) < 5
  is_invalid = cnn_test or small_q_test #or q_test

  return is_invalid 

def save_results(dev_path, 
  shared_path,
  gen_questions_path, 
  gen_answers_start_path,
  gen_answers_end_path,
  gen_idxs_path,
  gen_ids_path,
  save_path):
  print("Loading dev json: %s and shared: %s" % (dev_path, shared_path))
  dev_json = json.load(open(dev_path))
  shared_json = json.load(open(shared_path))
  print("Done loading dev json and shared")
  questions = utils.read_lines(gen_questions_path)
  answer_starts = utils.read_lines(gen_answers_start_path)
  answer_ends = utils.read_lines(gen_answers_end_path)
  idxs = utils.read_lines(gen_idxs_path)
  ids = utils.read_lines(gen_ids_path)

  keys = dev_json.keys()
  dataset = defaultdict(list)

  idx = 54

  for i in range(0, len(questions)):
    cur_q = questions[i].split(" ")
    if invalid_question(cur_q):
      continue 
    cur_q = dedup(cur_q)
    cur_ans_start = int(answer_starts[i])
    cur_ans_end = int(answer_ends[i])
    idx = int(idxs[i])
    id = int(ids[i])
    cur_par = shared_json['x'][idx][0][0]
    cy_0 = 0 
    cy_1 = len(cur_par[cur_ans_end - 1])
    cy = [[cy_0, cy_1]]

    answerss = [cur_par[cur_ans_start:cur_ans_end]]
    cur_q_char = list(map(lambda token: token.split(), cur_q))

    dataset['idxs'].append(idx)
    dataset['ids'].append(len(dataset['ids']))
    dataset['cy'].append(cy)
    dataset['answerss'].append(answerss)
    dataset['span_answerss'].append(answerss)
    dataset['*x'].append([idx, 0])
    dataset['*cx'].append([idx, 0])
    dataset['*p'].append([idx, 0])

    shared_json['x'][idx]
    dataset['y'].append([[[0, cur_ans_start], [0, cur_ans_end]]])
    dataset['q'].append(cur_q)
    dataset['cq'].append(cur_q_char)

  print("Saving to path %s" % save_path)
  utils.save_json(dataset, save_path)

save_directory = 'newsqa_unsupervised'
utils.check_dir(save_directory)

shared_path='newsqa/shared_train.json'
dev_path= 'newsqa/data_train.json'
base_path = '../datasets/newsqa_unsupervised/train'
gen_questions_path = '../logs/newsqa_saved_data/train_predictions_epoch_6.txt'#'%s/outputs.txt' % base_path#, 'newsqa/', 'newsqa/']
gen_answers_start_path = '%s/answer_starts.txt' % base_path
gen_answers_end_path = '%s/answer_ends.txt' % base_path 
gen_ids_path = '%s/ids.txt' % base_path 
gen_idxs_path = '%s/indices.txt' % base_path 
save_path = '%s/data_train.json' % save_directory

save_results(dev_path=dev_path, 
  shared_path=shared_path,
  gen_questions_path=gen_questions_path,
  gen_answers_start_path=gen_answers_start_path, 
  gen_answers_end_path=gen_answers_end_path,
  gen_ids_path=gen_ids_path,
  gen_idxs_path=gen_idxs_path,
  save_path=save_path)


"""
dev_paths = []#'newsqa/data_train.json', 'newsqa/data_dev.json', 'newsqa/data_test.json'] #'data/squad/data_train.json'
save_paths = ['newsqa_gen_filtered_v2/data_train.json']#'newsqa_gen/data_train.json', 'newsqa_gen/data_dev.json', 'newsqa_gen/data_test.json'] #'data/squad/web_data_train.json'
shared_paths = ['newsqa/shared_train.json']

#json_data = json.load(open(save_paths[0]))
#shared_data = json.load(open(shared_paths[0]))
#original_data = json.load(open(dev_paths[0]))

print(shared_data.keys())
for idx in range(100, 101): 
  print(json_data['q'][idx])
  print(original_data['q'][idx])
  print(original_data['answerss'][idx])
  print(json_data['answerss'][idx])


for dev_path, gen_questions_path, save_path in zip(dev_paths, gen_questions_paths, save_paths):
  save_results(dev_path, gen_questions_path, save_path)


"""