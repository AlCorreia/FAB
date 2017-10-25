from collections import Counter
import json
import numpy as np
import nltk
import math
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
                word2vec_dict[word] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict

def create_char2vec(config):
    glove_path = os.path.join(config['glove']['dir'], "glove.{}.{}d.txt".format(config['glove']['corpus'], config['glove']['vec_size']))
    vectors = {}
    with open(glove_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]

            for char in word:
                if ord(char) < 128:
                    if char in vectors:
                        vectors[char] = (vectors[char][0] + vec,
                                         vectors[char][1] + 1)
                    else:
                        vectors[char] = (vec, 1)

    base_name = os.path.splitext(os.path.basename(glove_path))[0] + '-char.txt'
    with open(os.path.join(config['glove']['dir'], base_name), 'w') as f2:
        for word in vectors:
            avg_vector = np.round(
                (vectors[word][0] / vectors[word][1]), 6).tolist()
            f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")


def get_char2vec(config, char_counter):
    char_path = os.path.join(config['glove']['dir'], "glove.{}.{}d-char.txt".format(config['glove']['corpus'], config['glove']['vec_size']))
    # If there is no pre-trained char embedding file, create one
    if os.path.isfile(char_path) is False:
        create_char2vec(config)
    char2vec_dict = {}
    with open(char_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(" ")
            char = array[0]
            vector = list(map(float, array[1:]))
            if char in char_counter:
                char2vec_dict[char] = vector

    return char2vec_dict


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
        tokenizer_result = [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
        return tokenizer_result


    source_path = in_path or os.path.join(config['directories']['source_dir'], "{}-v1.1.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_len = []
    paragraph_len, question_len = [], []
    word_counter, char_counter, word_counter_lower = Counter(), Counter(), Counter()
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
            max_xc = max([len(list(xij)) for xij in xi])
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                word_counter[xij] += len(para['qas'])
                word_counter_lower[xij.lower()]+=len(para['qas'])
                for xijk in xij:
                    char_counter[xijk] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                cqi = [list(qij) for qij in qi]
                max_qc = max([len(list(qij)) for qij in qi])
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
                    word_counter_lower[qij.lower()]+=1
                    word_counter[qij] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1
                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                paragraph_len.append(len(xi))
                word_len.append(max([max_qc, max_xc]))
                question_len.append(len(qi))
                answerss.append(answers)

            # TODO: Add debug option as in the original code
            # if args.debug:
            #     break

    if config['glove']['corpus'] == '6B':
        word2vec_dict = get_word2vec(config, word_counter_lower)
    else:
        word2vec_dict = get_word2vec(config, word_counter)
    char2vec_dict = {}
    if config['model']['pre_trained_char']:
        char2vec_dict = get_char2vec(config, char_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, 'paragraph_len': paragraph_len, 'question_len': question_len, 'word_len': word_len}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'word_counter_lower': word_counter_lower, 'char_counter': char_counter,
              'word2vec': word2vec_dict, 'char2vec': char2vec_dict}

    print("saving ...")
    save(config, data, shared, out_name)


def read_data(config, data_type, data_filter=None, data_train=None):
    data_path = os.path.join(config['directories']['dir'], "data_{}.json".format(data_type))
    shared_path = os.path.join(config['directories']['dir'], "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)


    num_examples = len(next(iter(data.values()))) #number of questions
    #max_par = max([len(shared['x'][i][j]) for i in range(len(shared['x'])) for j in range(len(shared['x'][i]))]) #max paragraph size
    #max_q = max([len(data['q'][i]) for i in range(len(data['q']))]) # max question size

    if data_filter: #if there is a filter to discard some of the passages or questions
        valid_idxs, valid_idxs_grouped = data_filter_func(config, data, shared)
    else:
        valid_idxs, valid_idxs_grouped = range(num_examples), [range(num_examples)]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))
    word2vec_dict = shared['word2vec']
    if data_train is not None: #dev is going to use the same unk chars and words. It changes only its known words from word2vec.
        #CHAR EMBEDDING
        shared['unk_char2idx'] = data_train['shared']['unk_char2idx']
        shared['known_char2idx'] = data_train['shared']['known_char2idx'] if config['model']['pre_trained_char'] else {}
        shared['emb_mat_known_chars'] = data_train['shared']['emb_mat_known_chars'] if config['model']['pre_trained_char'] else {}
        #WORDS EMBEDDING
        shared['unk_word2idx'] = data_train['shared']['unk_word2idx']
        shared['known_word2idx'] = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['unk_word2idx'])}
        known_idx2vec_dict = {idx: word2vec_dict[word] for word, idx in shared['known_word2idx'].items()}
        shared['emb_mat_known_words'] = np.array([known_idx2vec_dict[idx] for idx in range(len(known_idx2vec_dict))], dtype='float32')
    else:
        char2vec_dict = shared['char2vec']
        if config['glove']['corpus']=='6B':
            word_counter = shared['word_counter_lower'] #LOWER CASE
        else:
            word_counter = shared['word_counter']
        char2vec_dict = {}
        number_of_unk = config['pre']['number_of_unk']
        number_of_totens = config['model']['number_of_totens']
        char_counter = shared['char_counter']
    #WORD PRE-PROCESSING
        if config['pre']['finetune']: #false
            shared['unk_word2idx'] = {word: idx + 1 + number_of_unk + number_of_totens for idx, word in
                              enumerate(word for word, count in word_counter.items()
                                            if count > config['pre']['word_count_th'] or (config['pre']['known_if_glove'] and word in word2vec_dict))}
        else:
            assert config['pre']['known_if_glove']
            assert config['pre']['use_glove_for_unk']
            shared['unk_word2idx'] = {word: idx + 1+number_of_unk + number_of_totens for idx, word in #add 2 to UNK and NULL
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config['pre']['word_count_th'] and word not in word2vec_dict)} #threshold =10

        #CHAR PRE-PROCESSING
        if config['model']['pre_trained_char']:
            shared['unk_char2idx'] = {char: idx + 2 for idx, char in
                                      enumerate(char for char, count in char_counter.items()
                                                if count > config['pre']['char_count_th'] and char not in char2vec_dict)} #threshold =50
        else:
            shared['unk_char2idx'] = {char: idx + 2 for idx, char in
                                      enumerate(char for char, count in char_counter.items()
                                                if count > config['pre']['char_count_th'])} # threshold =50

        NULL = "-NULL-"
        UNK = "-UNK-"
        TOTEN = "-TOTEN-"
        shared['unk_word2idx'][NULL] = 0
        for i in range(number_of_unk):
            shared['unk_word2idx'][UNK+str(i)] = i+1
        for i in range(number_of_totens):
            shared['unk_word2idx'][TOTEN+str(i)] = i + (number_of_unk+1)
        shared['unk_char2idx'][NULL] = 0
        shared['unk_char2idx'][UNK] = 1

        if config['pre']['use_glove_for_unk']:
            # create word2idx for uknown and known words
            word_vocab_size=len(shared['unk_word2idx']) #vocabulary size of unknown words
            word2vec_dict = shared['word2vec']

            shared['known_word2idx'] = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['unk_word2idx'])}

            known_idx2vec_dict = {idx: word2vec_dict[word] for word, idx in shared['known_word2idx'].items()}
            # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))

            shared['emb_mat_known_words'] = np.array([known_idx2vec_dict[idx] for idx in range(len(known_idx2vec_dict))], dtype='float32')

            unk_idx2vec_dict = {idx: word2vec_dict[word] for word, idx in shared['unk_word2idx'].items() if word in word2vec_dict }

        if config['model']['char_embedding']:  # If there is char embedding
            if config['model']['pre_trained_char']:
                char_vocab_size = len(shared['unk_char2idx'])  # vocabulary size of unknown chars
                char2vec_dict = shared['char2vec']

                shared['known_char2idx'] = {char: idx for idx, char in enumerate(char for char in char2vec_dict.keys() if char not in shared['unk_char2idx'])}

                known_char_idx2vec_dict = {idx: char2vec_dict[char] for char, idx in shared['known_char2idx'].items()}
                # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))

                shared['emb_mat_known_chars'] = np.array([known_char_idx2vec_dict[idx] for idx in range(len(known_char_idx2vec_dict))], dtype='float32')

                unk_char_idx2vec_dict = {idx: char2vec_dict[word] for char, idx in shared['unk_char2idx'].items() if char in char2vec_dict }
                shared['emb_mat_unk_chars'] = np.array([unk_char_idx2vec_dict[idx] if idx in unk_char_idx2vec_dict
                                else np.random.multivariate_normal(np.zeros(int(config['model']['char_embedding_size'])),
                                                                   config['model']['variance_char_init']*np.eye(int(config['model']['char_embedding_size'])))
                                                       for idx in range(char_vocab_size)]) #create random vectors for new words
            else:
                char_vocab_size = len(shared['unk_char2idx'])
                shared['known_char2idx'] = {}

    data_set={'data':data, 'type':data_type, 'shared':shared, 'valid_idxs':valid_idxs, 'valid_idxs_grouped': valid_idxs_grouped}
    return data_set


def data_filter_func(config, data, shared):
    valid_idxs = []
    x_len = data['paragraph_len']
    q_len = data['question_len']
    word_len = data['word_len']
    # Delete paragraphs and questions longer than threshold.
    for i in range(len(data['q'])):
        #q = data['q'][i]
        #rx = data['*x'][i]
        #x = shared['x'][rx[0]][rx[1]]
        if (q_len[i] <= config['pre']['max_question_size']) and (x_len[i] <= config['pre']['max_paragraph_size']) and (word_len[i]<=config['pre']['max_word_size']):
            valid_idxs.append(i)

    # Group paragraphs and questions with similar sizes
    n_valid_idxs = len(valid_idxs)
    x_len = [x_len[valid_idxs[i]] for i in range(len(valid_idxs))]
    ordered_valid_idxs = [i[0] for i in sorted(zip(valid_idxs,x_len), key=lambda x:x[1])]
    # n_chunks: number of groups, in which questions are separated according to their paragraph sizes
    n_chunks = config['pre']['n_chunks']
    # Remainder must be taken into account, as n_valid_idxs/n_chunks might not divide evenly.
    remainder = n_valid_idxs % n_chunks
    chunk_size = math.floor(n_valid_idxs/n_chunks)
    valid_idxs_grouped = []
    index = 0
    # Compute the n_chunks groups of questions
    # TODO: I guess there might be a more elegant way than using this for if else
    for i in range(n_chunks):
        if remainder > 0:
            sub_list = ordered_valid_idxs[index:index+chunk_size+1]
            index = index + chunk_size + 1
            remainder = remainder - 1
        else:
            sub_list = ordered_valid_idxs[index:index+chunk_size]
            index = index+chunk_size
        valid_idxs_grouped.append(sub_list)
    # Check if the number of questions after grouping them is the same as before
    assert sum([len(valid_idxs_grouped[i]) for i in range(len(valid_idxs_grouped))]) == len(valid_idxs)
    # Return individual questions order by their respective paragraph size and grouped questions
    return ordered_valid_idxs, valid_idxs_grouped


def update_config(config, data_set):
    config['model']['vocabulary_size'] = len(data_set['shared']['unk_word2idx'])
    if config['model']['char_embedding']:
        config['model']['char_vocabulary_size'] = len(data_set['shared']['unk_char2idx'])
    return config


def get_batch_idxs(config, data_set, total_sort=False):
    valid_idxs_groups = data_set['valid_idxs_grouped']
    if total_sort:
        # Join all groups
        valid_idxs = [idxs for valid_idxs_group in valid_idxs_groups for idxs in valid_idxs_group]
    else:
        # Compute number of subgroups (separated by paragraph sizes)
        nGroups = len(valid_idxs_groups)
        # Choose randomly subgroup of questions (separated by paragraph sizes)
        group_choice = randint(0, nGroups-1)
        valid_idxs = valid_idxs_groups[group_choice]
    # Compute number of questions
    nQuestions = len(valid_idxs)
    n = 0
    batch_idxs = set()
    # Choose randomly batch_size questions from the selected subgroup
    while n < config['train']['batch_size']:
        batch_idxs.add(randint(0, nQuestions-1)) # It won't repeat IDs, because it is a set
        n = len(batch_idxs)
    batch_idxs = [valid_idxs[i] for i in batch_idxs]
    batch_idxs.sort()
    return batch_idxs
