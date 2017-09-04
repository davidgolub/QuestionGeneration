from helpers import utils
from helpers import spacy_tokenizer
import numpy as np 

delimiter='*@#$*($#@*@#$'
def func(l):
    items = l.strip().split(delimiter)
    return items

def gen_func(l):
    items = l.strip().split(" ")
    return items 

answer_starts_path = 'datasets/newsqa/train/answer_starts.txt'
answer_ends_path = 'datasets/newsqa/train/answer_ends.txt'
input_path = 'datasets/newsqa/train/inputs.txt'
output_path = 'datasets/newsqa/train/outputs.txt'
generated_path = 'logs/newsqa_saved_data/dummy5_train_predictions_epoch_6.txt'
indices_path = 'datasets/newsqa/train/indices.txt'


inputs = utils.read_lines_with_func(func, input_path)
outputs = utils.read_tabbed_lines(output_path)
generated = utils.read_lines_with_func(gen_func, generated_path)
answer_starts = list(map(lambda l: int(l), utils.read_lines(answer_starts_path)))
answer_ends = list(map(lambda l: int(l), utils.read_lines(answer_ends_path)))
indices = list(map(lambda l: int(l), utils.read_lines(indices_path)))

answers = []
truncated_contexts = []
questions = []
generated_questions = []

num_overlap = []
num_items = len(generated)

question_counter = 0 
generated_question_counter = 0 
filtered_words = ["a", "the", "who", "what", "when", "where", "why", "it"]
for i in range(num_items):
    start_idx = answer_starts[i]
    end_idx = answer_ends[i]
    idx = indices[i]
    padded_start_idx = np.max([0, start_idx-10])
    padded_end_idx = np.min([end_idx + 10, len(inputs[idx])])
    truncated_context = inputs[idx][padded_start_idx:padded_end_idx]
    
    answers.append(inputs[idx][start_idx:end_idx])
    truncated_contexts.append(truncated_context)

    question = outputs[i]
    generated_question = generated[i]

    questions.append(question)
    generated_questions.append(generated_question)

    for t in question:
        if t not in filtered_words:
            if t in truncated_context:
                question_counter += 1 

    for t in generated_question:
        if t not in filtered_words:
            if t in truncated_context: 
                generated_question_counter += 1 


    #ner_tokens = spacy_tokenizer.extract_NER(' '.join(truncated_context))
    #assert(False)

utils.save_tabbed_lines(questions, "analysis/questions.txt")
utils.save_tabbed_lines(generated_questions, "analysis/generated_questions.txt")
utils.save_tabbed_lines(answers, "analysis/answers.txt")
utils.save_tabbed_lines(truncated_contexts, "analysis/truncated_contexts.txt")

num_tokens_q =  question_counter / float(num_items)
num_tokens_generated_q = generated_question_counter / float(num_items) 


print(num_tokens_q)
print(num_tokens_generated_q)

1.8647080433936984
2.5078769935600986

2.958032588494619
4.5196546656869945

2.469334831654925
4.094059298958379
#utils.save_lines()
#utils.save_lines():

#If you look at the fraction of questions that have: overlapping words with the context
#Of question words that overlap with the context from synthetically generated questions
#Of question words that overlap with the context from human-generated words.








