newsqa_import json 
from squad.utils import get_2d_spans
from helpers import utils
from helpers import spacy_tokenizer

def create_dataset(save_dir, data_path, shared_path):
    print("Loading data from path %s" % data_path)
    data = json.load(open(data_path))
    print("Done loading data")
    shared_data = json.load(open(shared_path))
    print("Done loading shared data from path %s" % shared_path)

    def count_sums(up_to_idx):
        total_len = 0
        for i in range(0, up_to_idx):
            total_len += len(shared_data['x'][i])
        return total_len

    idxs = []
    xs = [] 
    answer_starts = [] 
    answer_ends = []
    indices = []
    questions = []

    for i in range(len(shared_data['x'])):
        print("On %s of %s" % (i, len(shared_data['x'])))
        for j in range(len(shared_data['x'][i])):
            cur_tokens = shared_data['x'][i][j][0]
            cur_text = " ".join(cur_tokens)
            cur_ans_starts, cur_ans_ends = spacy_tokenizer.extract_phrases(cur_text, 2)
            answer_starts.extend([str(ans) for ans in cur_ans_starts])
            answer_ends.extend([str(ans) for ans in cur_ans_ends])
            idxs.extend(range(len(idxs), len(idxs) + len(cur_ans_starts)))
            questions.extend(["<NONE>"] * len(cur_ans_starts))
            indices.extend([str(len(xs))] * len(cur_ans_starts))
            xs.append('\t'.join(cur_tokens))

    idxs = list(map(lambda idx: str(idx), idxs))
    utils.save_lines(idxs, '%s/ids.txt' % save_dir)
    utils.save_lines(questions, '%s/outputs.txt' % save_dir)
    utils.save_lines(answer_starts, '%s/answer_starts.txt' % save_dir)
    utils.save_lines(answer_ends, '%s/answer_ends.txt' % save_dir)
    utils.save_lines(xs, '%s/inputs.txt' % save_dir)
    utils.save_lines(indices, '%s/indices.txt' % save_dir)


# Create squad dataset
create_dataset(save_dir='../datasets/newsqa_unsupervised/',
data_path='newsqa/data_train.json',
shared_path='newsqa/shared_train.json')