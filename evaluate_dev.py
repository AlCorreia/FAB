""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import numpy as np
import pdb
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def update_statistics(EM_F1, classifier, statistics):
    EM, F1 = EM_F1
    for i in classifier.keys():
        statistics[i]['n'][classifier[i]] = statistics[i]['n'][classifier[i]]  + 1
        statistics[i]['EM'][classifier[i]] = statistics[i]['EM'][classifier[i]]  + EM
        statistics[i]['F1'][classifier[i]] = statistics[i]['F1'][classifier[i]]  + F1
    return statistics

def create_statistics():
    par_len_size = 14
    q_type_size = 11
    q_len_size = 10
    ans_size = 21
    statistics = {'par_len':{'n':np.zeros(par_len_size), 'EM':np.zeros(par_len_size), 'F1':np.zeros(par_len_size)},
               'q_type': {'n':np.zeros(q_type_size), 'EM':np.zeros(q_type_size), 'F1':np.zeros(q_type_size)},
               'q_len': {'n':np.zeros(q_len_size), 'EM':np.zeros(q_len_size), 'F1':np.zeros(q_len_size)},
               'ans_len': {'n':np.zeros(ans_size), 'EM':np.zeros(ans_size), 'F1':np.zeros(ans_size)}}
    return statistics

def question_classifier(data, index):
    classifier = {}
    #type of question
    classifier['q_type'] = data['question_type'][index]
    #size of passage
    par_len = data['paragraph_len'][index]
    par_lens = [30,60,90,120,150,180,210,240,270,300,330,360,390]
    for i in range(len(par_lens)):
        if par_len <=par_lens[i]:
            classifier['par_len'] = i
            break
    if not ('par_len' in classifier):
         classifier['par_len'] = len(par_lens) #longer than 390
    #size of question
    q_len = data['question_len'][index]
    q_lens = [5,10,15,20,25,30,35,40,45]
    for i in range(len(q_lens)):
        if q_len <=q_lens[i]:
            classifier['q_len'] = i
            break
    if not ('q_len' in classifier):
         classifier['q_len'] = len(q_lens) #longer than 390
    #size of answer
    answer = data['y'][i]
    f = lambda x: x[1]-x[0]+1
    answer_len = min(list(map(f,answer)))
    answer_lens = list(range(20))
    for i in range(len(answer_lens)):
        if answer_len <=answer_lens[i]:
            classifier['ans_len'] = i
            break
    if not ('ans_len' in classifier):
         classifier['ans_len'] = len(answer_lens) #longer than 390
    return classifier

def evaluate(dataset, predictions):
    statistics = create_statistics()
    f1 = exact_match = total = 0
    total = len(dataset['data']['q'])
    for i in range(len(dataset['data']['q'])):
        classifier = question_classifier(dataset['data'], i)
        ground_truths = dataset['data']['answerss'][i]
        prediction = predictions[dataset['data']['ids'][i]]
        EM_i = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        exact_match += EM_i
        f1_i = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_i
        statistics = update_statistics([EM_i,f1_i], classifier, statistics)
    pdb.set_trace()
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def evaluate_dev(dataset, predictions):
    exact_match, f1 = evaluate(dataset, predictions)
    return  exact_match, f1
