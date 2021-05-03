"""
Prepares data pickle files for train/val/test splits (for 5-folds)
"""
import numpy as np
import argparse
import os
import itertools
import logging
import pickle

def parse_args():
    parser = argparse.ArgumentParser('preparation of (train-val) letor dataset pickle files')
    parser.add_argument('--fold', default=1, type=int, help='which fold to be run')
    parser.add_argument('--nr_folds', default=5, type=int, help='number of cross validation folds')
    parser.add_argument('--cv', action="store_true", default=False, help='if would like to apply cross-validation from fold 1 to nr_folds')
    parser.add_argument('parent_dir', help='path to the dataset (e.g. MQ/MQ2007/)')
    parser.add_argument('--out_dir', default='../data/MQ/MQ2007_pkl/', help='path to data pickles to be saved')
    parser.add_argument('--fea_dim', default=46, type=int, help='LETOR ranking feature size per document query pair')
    return parser.parse_args()


class LETORLine:
    def __init__(self, line, num_features=46):
        if '#' in line:
            line, line_info = line.split("#")
            self.document_info = line_info.rstrip('\n')
        else:
            self.document_info = ""
        line_split = line.split()
        self.label = int(line_split[0])
        self.query_id = line_split[1].replace('qid:', '')
        self.features = np.empty(num_features)
        for fea_idx in range(num_features):
            self.features[fea_idx] = float(line_split[2 + fea_idx].split(':')[1])


class LETORData:
    def __init__(self, lines, num_features=46):
        self.lines = lines
        self.fea_dim = num_features
        self.num_instances = len(lines)
        self.features = np.zeros((self.num_instances, num_features))
        self.labels = np.zeros(self.num_instances, dtype=np.uint8)
        self.queries = []
        self.loop_over_lines()
        self.gather_listwise_data()
    #
    def loop_over_lines(self):
        for (l_idx, line_i) in enumerate(self.lines):
            line = LETORLine(line_i, num_features=self.fea_dim)
            self.features[l_idx, :] = line.features
            self.queries.append(line.query_id)
            self.labels[l_idx] = line.label
    #
    def gather_listwise_data(self):
        nr_queries = len(set(self.queries))
        self.listwise_num_docs_per_queries = [len(list(g)) for k, g in itertools.groupby(self.queries)]
        assert nr_queries == len(self.listwise_num_docs_per_queries)
        max_nr_docs = max(self.listwise_num_docs_per_queries)
        logging.info('max number of document for a query :{}'.format(max_nr_docs))
        logging.info('mean number of document for a query :{}'.format(np.mean(self.listwise_num_docs_per_queries)))
        logging.info('median number of document for a query :{}'.format(np.median(self.listwise_num_docs_per_queries)))
        logging.info('q3 of number of document for a query :{}'.format(np.quantile(self.listwise_num_docs_per_queries, 0.75)))
        logging.info('number of queries :{}'.format(nr_queries))
        logging.info('total number of instances :{}'.format(self.num_instances))
        self.listwise_features = np.zeros((nr_queries, max_nr_docs, self.fea_dim))
        self.listwise_query_ids = []
        self.listwise_labels = np.zeros((nr_queries, max_nr_docs), dtype=np.uint8)
        doc_idx = 0
        for (idx, nr_docs) in enumerate(self.listwise_num_docs_per_queries):
            assert len(set(self.queries[doc_idx:doc_idx+nr_docs])) ==1
            self.listwise_query_ids.append(self.queries[doc_idx])
            self.listwise_labels[idx, :nr_docs] = self.labels[doc_idx:doc_idx+nr_docs]
            self.listwise_features[idx, :nr_docs, :] = self.features[doc_idx:doc_idx+nr_docs, :]
            doc_idx += nr_docs


def main(args):
    data_parts = ['train', 'vali', 'test']
    # create directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # set logger
    logging.basicConfig(filename=os.path.join(args.out_dir, 'data_log.txt'), filemode='a', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # loop over different folds 1 to 5 for LETOR datasets
    for fold_i in range(args.fold, args.fold+args.nr_folds):
        fold_dir = os.path.join(args.parent_dir, 'Fold{}'.format(fold_i))
        outfold_dir = os.path.join(args.out_dir, 'Fold{}'.format(fold_i))
        # create directory if it doesn't exist
        if not os.path.exists(outfold_dir):
            os.makedirs(outfold_dir)
        for data in data_parts:
            logging.info('Starting fold{} {}'.format(fold_i, data))
            with open(os.path.join(fold_dir, '{}.txt'.format(data)), 'r') as f:
                train_lines = f.readlines()
            train_data = LETORData(train_lines, num_features=args.fea_dim)
            train_dict = {'x':train_data.listwise_features, 'y':train_data.listwise_labels,
                          'q_ids':train_data.listwise_query_ids, 'num_docs':train_data.listwise_num_docs_per_queries}
            pickle.dump(train_dict, open(os.path.join(outfold_dir, '{}.pkl'.format(data)), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('-done.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

