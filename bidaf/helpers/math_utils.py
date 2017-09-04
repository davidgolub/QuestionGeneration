import numpy as np

def top_k_spans(start_probs, end_probs, n, k):
    """
    Returns top k non overlapping spans for a passage
    sorted by start/end probabilities
    """
    probs = []
    argmax_spans = []
    for i in range(k + 1):
        probs.append([])
        argmax_spans.append([])
        for j in range(n + 1):
            probs[i].append(0)
            argmax_spans[i].append([-1, -1])

    for i in range(k + 1):
        probs[i][0] = 0;
 
    for j in range(n + 1):
        probs[0][j] = 0

    # fill the table in bottom-up fashion
    for i in range(1, k + 1):
        prev_diff = -10000
        prev_idx = -1
        for j in range(1, n):
            if prev_diff < probs[i-1][j-1] - start_probs[j-1]:
                prev_diff = probs[i-1][j-1] - start_probs[j-1]
                prev_idx = j-1
            if probs[i][j-1] > end_probs[j] + prev_diff:
                probs[i][j] = probs[i][j-1]
                argmax_spans[i][j] = argmax_spans[i][j-1]
            else:
                probs[i][j] = end_probs[j] + prev_diff
                argmax_spans[i][j] = (prev_idx, j)

    max_probs = probs[k][n-1]
    cur_probs = max_probs
    cur_spans = argmax_spans[k][n-1]
    start_end_idxs = []
    start_end_probs = []

    while cur_probs > 0:
        cur_indices = cur_spans
        cur_prob = end_probs[cur_indices[1]] - start_probs[cur_indices[0]]
        start_end_probs.append(cur_prob)
        cur_probs = cur_probs - cur_prob
        start_end_idxs.append(cur_indices)    
        cur_spans = argmax_spans[k][cur_indices[0]]

    return max_probs, start_end_idxs, start_end_probs