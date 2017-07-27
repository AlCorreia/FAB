from collections import Counter
import json
import numpy as np
import nltk
# nltk.download('punkt') Uncomment if you haven't installed the package yet
import pandas as pd
import os
import pdb
from random import randint
from tqdm import tqdm
from utils import get_word_span, process_tokens


def get_word2vec(config, word_counter):
    glove_path = os.path.join(config['glove']['dir'], "glove.{}.{}d.txt".format(config['glove']['corpus'], config['glove']['vec_size']))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[config['glove']['corpus']]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def save(config, data, shared, data_type):
    """
        Save json files for dictionaries data and shared to the directory
        specified in config.
    """
    data_path = os.path.join(config['directories']['target_dir'], "data_{}.json".format(data_type))
    shared_path = os.path.join(config['directories']['target_dir'], "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def prepro_each(config, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    """
        Pre-process the whole dataset and create two dictionaries with the
        tokens of each example.
        The dictionaries are dumped into json files.

        Dictionaries structure:
        - x: passages split in tokens.
        - cx: passages split in characters.
        - rx: reference to the article and paragraph correspondent to each question (index)
        - q: questions split in words.
        - cq: questions split in characters
        - y: tuple with start and end indices of the word tokens
        - cy: tuple with start and end indices of the character tokens
        - ids: the unique id in squad
        - idxs: incremental id of each quesiton
        - answerss: the answer text

    """
    # Define the word tokenizer. Only NLTK for now
    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


    source_path = in_path or os.path.join(config['directories']['source_dir'], "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = word_tokenize(context)
            xi = process_tokens(xi)  # process tokens
            # given xi, add chars
            cxi = [list(xij) for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                word_counter[xij] += len(para['qas'])
                lower_word_counter[xij.lower()] += len(para['qas'])
                for xijk in xij:
                    char_counter[xijk] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    yi0, yi1, i0, i1 = get_word_span(context, xi, answer_start, answer_stop)
                    w0 = xi[yi0]
                    w1 = xi[yi1-1]
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1 #in case the answer_end does not correspond to the end of a word from tokenizer
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0) #check if first character matches
                    assert answer_text[-1] == w1[cyi1] #check if last character matches
                    assert cyi0 < 32, (answer_text, w0) #why?
                    assert cyi1 < 32, (answer_text, w1) #why?

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])
                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1
                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

            # TODO: Add debug option as in the original code
            # if args.debug:
            #     break

    word2vec_dict = get_word2vec(config, word_counter)
    lower_word2vec_dict = get_word2vec(config, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(config, data, shared, out_name)


def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config['directories']['out_dir'], "data_{}.json".format(data_type))
    shared_path = os.path.join(config['directories']['out_dir'], "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)
        
    num_examples = len(next(iter(data.values()))) #number of questions
    if data_filter is None: #if there is a filter to discard some of the passages or questions
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values): #each question
            each = {key: val for key, val in zip(keys, vals)} #rebuilt dictionary for each question
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))
    if not ref: #TODO: why there is this "if"? If not useful, delete
        word2vec_dict = shared['lower_word2vec'] if config['pre']['lower_word'] else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config['pre']['lower_word'] else shared['word_counter']
        char_counter = shared['char_counter']
        if config['pre']['finetune']: #false
            shared['unk_word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config['pre']['word_count_th'] or (config['pre']['known_if_glove'] and word in word2vec_dict))}
        else:
            assert config['pre']['known_if_glove']
            assert config['pre']['use_glove_for_unk']
            shared['unk_word2idx'] = {word: idx + 2 for idx, word in #add 2 toUNK and NULL
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config['pre']['word_count_th'] and word not in word2vec_dict)} #threshold =10
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                      if count > config['pre']['char_count_th'])} #threshold =50
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['unk_word2idx'][NULL] = 0
        shared['unk_word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        #json.dump({'unk_word2idx': shared['unk_word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w')) why to use this line?
    else: 
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config['pre']['use_glove_for_unk']:
        # create word2idx for uknown and known words
        word_vocab_size=len(shared['unk_word2idx']) #vocabulary size of unknown words
        word2vec_dict = shared['lower_word2vec'] if config['pre']['lower_word'] else shared['word2vec']
        
        shared['known_word2idx'] = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['unk_word2idx'])}
        
        known_idx2vec_dict = {idx: word2vec_dict[word] for word, idx in shared['known_word2idx'].items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        
        shared['emb_mat_known_words'] = np.array([known_idx2vec_dict[idx] for idx in range(len(known_idx2vec_dict))], dtype='float32')
        
        unk_idx2vec_dict = {idx: word2vec_dict[word] for word, idx in shared['unk_word2idx'].items() if word in word2vec_dict }
        
        shared['emb_mat_unk_words']  = np.array([unk_idx2vec_dict[idx] if idx in unk_idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(int(config['glove']['vec_size'])), np.eye(int(config['glove']['vec_size'])))
                        for idx in range(word_vocab_size)]) #create random vectors for new words

    data_set={'data':data,'type':data_type,'shared':shared,'valid_idxs':valid_idxs}
    return data_set

def update_config(config, data_set):
    config['model']['vocabulary_size'] = len(data_set['shared']['emb_mat_unk_words'])
    config['model']['emb_mat_unk_words'] = np.array(data_set['shared']['emb_mat_unk_words'], dtype = np.float32)
    return config

def get_batch_idxs(config, data_set):
    # Compute number of questions
    nQuestions = len(next(iter(data_set['data'].values())))
    n = 0
    batch_idxs = set();
    while n < config['model']['batch_size']:
        batch_idxs.add(randint(0, nQuestions-1))
        n = len(batch_idxs)
    batch_idxs = list(batch_idxs)
    batch_idxs.sort()
    return batch_idxs
