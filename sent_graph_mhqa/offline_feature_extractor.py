import argparse
from envs import OUTPUT_FOLDER, DATASET_FOLDER, PRETRAINED_MODEL_FOLDER
from model_envs import MODEL_CLASSES
from sent_graph_mhqa.sent_graph_datahelper import DataHelper
import os
from os.path import join
import torch

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def load_encoder_model(encoder_name_or_path, model_type):
    if encoder_name_or_path in [None, 'None', 'none']:
        raise ValueError('no checkpoint provided for model!')

    config_class, model_encoder, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(encoder_name_or_path)
    if config is None:
        raise ValueError(f'config.json is not found at {encoder_name_or_path}')

    # check if is a path
    if os.path.exists(encoder_name_or_path):
        if os.path.isfile(os.path.join(encoder_name_or_path, 'pytorch_model.bin')):
            encoder_file = os.path.join(encoder_name_or_path, 'pytorch_model.bin')
        else:
            encoder_file = os.path.join(encoder_name_or_path, 'encoder.pkl')
        encoder = model_encoder.from_pretrained(encoder_file, config=config)
    else:
        encoder = model_encoder.from_pretrained(encoder_name_or_path, config=config)

    return encoder, config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir',
                        type=str,
                        default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument("--dev_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_dev_distractor_v1.json'))
    parser.add_argument("--train_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_train_v1.1.json'))

    # model
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--encoder_name_or_path",
                        default='roberta-large',
                        type=str,
                        # help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
                        help="Path to pre-trained model or shortcut name selected")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size",
                        # default=8,
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int)
    # encoder
    parser.add_argument("--frozen_layer_number", default=0, type=int)
    parser.add_argument("--fine_tuned_encoder", default=None, type=str)
    parser.add_argument("--fine_tuned_encoder_path", default=PRETRAINED_MODEL_FOLDER, type=str)

    # train-dev data type
    parser.add_argument("--daug_type", default='long_low', type=str, help="Train Data augumentation type.")
    parser.add_argument("--devf_type", default='long_low', type=str, help="Dev data type")

    # eval
    parser.add_argument("--encoder_ckpt", default=None, type=str)
    parser.add_argument("--model_ckpt", default=None, type=str)

    ##################################
    # learning and log
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    # hyper-parameter
    parser.add_argument("--max_para_num", default=4, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_entity_num", default=60, type=int)


def feature_extraction(args):
    encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
    encoder_path = join(args.fine_tuned_encoder_path, args.fine_tuned_encoder, 'encoder.pkl')
    print("Loading encoder from: {}".format(encoder_path))
    encoder.load_state_dict(torch.load(encoder_path))
    print("Loading encoder completed")

    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path, do_lower_case=True)
    sep_token_id = tokenizer.sep_token_id
    data_helper = DataHelper(sep_token_id=sep_token_id, config=args)

    dev_data_loader = data_helper.hotpot_val_dataloader

    encoder.eval()
    for step, batch in enumerate(dev_data_loader):
        inputs = {'input_ids': batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert',
                                                                                        'xlnet'] else None}  # XLM don't use segment_ids
        outputs = encoder(**inputs)