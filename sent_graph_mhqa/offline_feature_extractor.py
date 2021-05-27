import argparse
from model_envs import MODEL_CLASSES
import gzip
import pickle
import os
from sent_graph_mhqa.sent_graph_datahelper import DataHelper
from sent_graph_mhqa.sg_utils import sent_state_feature_extractor
from utils.gpu_utils import single_free_cuda
from os.path import join
import torch
from tqdm import tqdm

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
                        required=True,
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
    parser.add_argument("--data_type", type=str, required=True)
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
    parser.add_argument("--save_batch_size", default=1000, type=int)
    parser.add_argument("--cpu_num", default=8, type=int)

    args = parser.parse_args()
    return args

def complete_default_train_parser(args):
    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    args.device = device
    graph_feature_output_folder = os.path.join(args.output_dir, 'graph')
    os.makedirs(graph_feature_output_folder, exist_ok=True)
    args.output_dir = graph_feature_output_folder
    args.max_doc_len = 512
    args.input_dim = 768 if 'base' in args.encoder_name_or_path else (4096 if 'albert' in args.encoder_name_or_path else 1024)
    return args

def feature_extraction(args):
    encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
    encoder_path = join(args.input_model_path, 'encoder.pkl')
    print("Loading encoder from: {}".format(encoder_path))
    encoder.load_state_dict(torch.load(encoder_path))
    print("Loading encoder completed")

    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path, do_lower_case=True)
    sep_token_id = tokenizer.sep_token_id
    data_helper = DataHelper(sep_token_id=sep_token_id, config=args)

    if 'train' in args.data_type:
        hotpot_data_loader = data_helper.hotpot_train_dataloader
        hotpot_example_dict = data_helper.train_example_dict
    else:
        hotpot_data_loader = data_helper.hotpot_val_dataloader
        hotpot_example_dict = data_helper.dev_example_dict

    encoder=encoder.to(args.device)
    encoder.eval()
    graph_features = []
    save_idx = 0
    total_graph_num = 0
    for step, batch in enumerate(tqdm(hotpot_data_loader)):
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

        batch_size = sent_representations.shape[0]
        supp_sent_np = batch['is_support'].cpu().numpy()
        sent_num_list = batch['sent_num'].cpu().numpy().tolist()
        sent_representations_np = sent_representations.detach().cpu().numpy()
        sent_mask_np = batch['sent_mask'].cpu().numpy()
        for idx in range(batch_size):
            key = batch['ids'][idx]
            edges = batch['edges'][idx]
            sent_num = sent_num_list[idx]
            assert sent_num > 0
            sent_embed = sent_representations_np[idx][:sent_num]
            sent_mask = sent_mask_np[idx]
            example_i = hotpot_example_dict[key]
            sent_names = example_i.sent_names
            supp_sent_labels = supp_sent_np[idx]
            graph_i = {'id': key, 'feat': sent_embed, 'num': sent_num, 'edge': edges, 'mask': sent_mask, 'name': sent_names, 'label': supp_sent_labels}
            graph_features.append(graph_i)
            if len(graph_features) % args.save_batch_size == 0:
                output_file_name = join(args.output_dir, 'graph_with_feature_{}.pickle'.format(save_idx))
                print('Graph example file name = {}'.format(output_file_name))
                with gzip.open(output_file_name, 'wb') as fout:
                    pickle.dump(graph_features, fout)
                print('Saving {} examples in {}'.format(len(graph_features), output_file_name))
                total_graph_num = total_graph_num + len(graph_features)
                save_idx = save_idx + 1
                graph_features = []
    if len(graph_features) > 0:
        output_file_name = join(args.output_dir, 'graph_with_feature_{}.pickle'.format(save_idx))
        print('Graph example file name = {}'.format(output_file_name))
        with gzip.open(output_file_name, 'wb') as fout:
            pickle.dump(graph_features, fout)
        print('Saving {} examples in {}'.format(len(graph_features), output_file_name))
        total_graph_num = total_graph_num + len(graph_features)
    print('Total graph number = {}/{}'.format(total_graph_num, save_idx))
    return graph_features

if __name__ == '__main__':
    args = parse_args()
    args = complete_default_train_parser(args=args)
    feature_extraction(args=args)