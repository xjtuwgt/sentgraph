import pickle
import torch
import json
import numpy as np
import string
import re
import os
import shutil
import collections
import logging
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from model_envs import MODEL_CLASSES


logger = logging.getLogger(__name__)

def load_encoder_model(encoder_name_or_path, model_type, cached_dir=None):
    if encoder_name_or_path in [None, 'None', 'none']:
        raise ValueError('no checkpoint provided for model!')

    config_class, model_encoder, _ = MODEL_CLASSES[model_type]
    if cached_dir is None:
        config = config_class.from_pretrained(encoder_name_or_path)
    else:
        config = config_class.from_pretrained(encoder_name_or_path, cache_dir=cached_dir)
    if config is None:
        raise ValueError(f'config.json is not found at {encoder_name_or_path}')

    # check if is a path
    # if os.path.exists(encoder_name_or_path):
    #     if os.path.isfile(os.path.join(encoder_name_or_path, 'pytorch_model.bin')):
    #         encoder_file = os.path.join(encoder_name_or_path, 'pytorch_model.bin')
    #     else:
    #         encoder_file = os.path.join(encoder_name_or_path, 'encoder.pkl')
    #     encoder = model_encoder.from_pretrained(encoder_file, config=config)
    # else:
    #     encoder = model_encoder.from_pretrained(encoder_name_or_path, config=config)
    encoder = model_encoder(config)

    return encoder, config



def convert_to_tokens(examples, features, ids, y1, y2, q_type_prob):
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    q_type = np.argmax(q_type_prob, 1)

    def get_ans_from_pos(qid, y1, y2):
        feature = features[qid]
        example = examples[qid]

        tok_to_orig_map = feature.token_to_orig_map
        orig_all_tokens = example.question_tokens + example.doc_tokens

        final_text = " "
        if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
            orig_tok_start = tok_to_orig_map[y1]
            orig_tok_end = tok_to_orig_map[y2]

            ques_tok_len = len(example.question_tokens)
            if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
                ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
                ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(example.question_tokens[orig_tok_end])
                final_text = example.question_text[ques_start_idx:ques_end_idx]
            else:
                orig_tok_start -= len(example.question_tokens)
                orig_tok_end -= len(example.question_tokens)
                ctx_start_idx = example.ctx_word_to_char_idx[orig_tok_start]
                ctx_end_idx = example.ctx_word_to_char_idx[orig_tok_end] + len(example.doc_tokens[orig_tok_end])
                final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[orig_tok_end]+len(example.doc_tokens[orig_tok_end])]

        return final_text

    for i, qid in enumerate(ids):
        feature = features[qid]
        answer_text = ''
        if q_type[i] in [0, 3]:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        else:
            raise ValueError("question type error")

        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

    return answer_dict, answer_type_dict, answer_type_prob_dict


def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights


def get_bias(size):
    bias = nn.Parameter(torch.zeros(size=size))
    return bias


def get_act(act):
    if act.startswith('lrelu'):
        return nn.LeakyReLU(float(act.split(':')[1]))
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError