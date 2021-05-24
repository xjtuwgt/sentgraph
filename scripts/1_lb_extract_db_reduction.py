import json
import pickle
import argparse
from time import time
from tqdm import tqdm

def load_link_ner_data(link_ner_pickle_file):
    print('Loading data from {}'.format(link_ner_pickle_file))
    start_time = time()
    link_ner_list = pickle.load(open(link_ner_pickle_file, "rb"))
    print('Mapping {} records takes {:.4f}'.format(len(link_ner_list), time() - start_time))

    # 1. map title to ID, link, ner
    title_to_id_link_ner = {}
    start_time = time()
    for link_ner in tqdm(link_ner_list):
        doc_id, title, hyperlink_titles, text_ner = link_ner
        if title not in title_to_id_link_ner:
            title_to_id_link_ner[title] = (doc_id, hyperlink_titles, text_ner)
    print('Mapping title to ID takes {:.4f}'.format(time() - start_time))
    return title_to_id_link_ner

def link_ner_pickle_extraction(input_file, link_ner_file):
    start_time = time()
    input_data = json.load(open(input_file, 'r'))
    print('Loading {} records from {} in {:.4f}'.format(len(input_data), input_file, time() - start_time))
    start_time = time()
    title_to_id_link_ner = load_link_ner_data(link_ner_pickle_file=link_ner_file)
    print('Loading {} records from {} in {:.4f}'.format(len(title_to_id_link_ner), link_ner_file, time() - start_time))

    output_data = {}
    for data in tqdm(input_data):
        context = dict(data['context'])
        for title in context.keys():
            if title not in title_to_id_link_ner:
                print("{} not exist in DB".format(title))
            else:
                doc_id, hyperlink_titles, text_ner  = title_to_id_link_ner[title]
                output_data[title] = {'hyperlink_titles': hyperlink_titles,
                                      'text_ner': text_ner}
    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--pickle_name", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_data = link_ner_pickle_extraction(input_file=args.input, link_ner_file=args.pickle_name)
    print('Generating {} records'.format(len(output_data)))
    output_file = args.output
    json.dump(output_data, open(output_file, 'w'))
    print('Saving {} records into {}'.format(len(output_data), output_file))