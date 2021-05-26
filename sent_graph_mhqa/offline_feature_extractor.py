import argparse
from model_envs import MODEL_CLASSES
from envs import OUTPUT_FOLDER
from sent_graph_mhqa.sent_graph_datahelper import DataHelper
from sent_graph_mhqa.sg_utils import sent_state_feature_extractor
from utils.gpu_utils import single_free_cuda
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
    encoder = model_encoder(config)
    return encoder, config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        type=str,
                        default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    # model
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)

    parser.add_argument("--encoder_name_or_path",
                        default='albert-xxlarge-v2',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected")
    parser.add_argument("--model_type", default='albert', type=str, help="alber reader model")
    parser.add_argument('--input_model_path', default=None, type=str, required=True)
    parser.add_argument("--encoder_ckpt", default='encoder.pkl', type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    # encoder
    parser.add_argument("--frozen_layer_number", default=0, type=int)
    parser.add_argument("--fine_tuned_encoder", default=None, type=str)

    # train-dev data type
    parser.add_argument("--daug_type", default='long_low', type=str, help="Train Data augumentation type.")
    parser.add_argument("--devf_type", default='long_low', type=str, help="Dev data type")

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

    args = parser.parse_args()
    return args

def complete_default_train_parser(args):
    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    args.device = device
    args.max_doc_len = 512
    args.input_dim = 768 if 'base' in args.encoder_name_or_path else (4096 if 'albert' in args.encoder_name_or_path else 1024)
    return args

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
        for key, value in batch.items():
            if key not in {'ids', 'edges'}:
                batch[key] = value.to(args.device)

        inputs = {'input_ids': batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
        with torch.no_grad():
            outputs = encoder(**inputs)
            context_emb = outputs[0]
            sent_representations = sent_state_feature_extractor(batch=batch, input_state=context_emb)
            print(sent_representations.shape)

if __name__ == '__main__':
    args = parse_args()
    feature_extraction(args=args)
