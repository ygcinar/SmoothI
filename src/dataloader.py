import torch
from torch.utils.data import Dataset
import logging
import numpy as np
import pickle


class LETORDatasetListwise(Dataset):
    """
    Dataset class - features, relevance label, info of query-level 'query-document pair features'
    """
    def __init__(self, pkl_file, device='cpu', return_info=False, rank_list_length=121):
        """
        Args:
            pkl_file: (string): Path to the data pkl file.
            device: (string) cpu or cuda device
            return_info: (boolean)
            rank_list_length: (int) max. number of documents for a query
        """
        data_dict = pickle.load(open(pkl_file, 'rb'))
        self.data_x = data_dict['x']  # query-document pair features matrix, .shape: (#ofqueries, max #ofdocuments, #offeatures)
        self.data_y = data_dict['y']  # relevance label matrix, .shape: (#ofqueries, max #ofdocuments)
        self.max_num_docs = self.data_x.shape[1]  # max. number of documents for a query
        self.data_qids = data_dict['q_ids']  # query ids
        self.data_num_docs = data_dict['num_docs']  # number of documents related to query, .shape: (#ofqueries, 1)
        self.device = device  # cpu, or cuda device
        self.return_info = return_info  # return query document info
        self.list_length = rank_list_length
        #
    #
    def __len__(self):
        return len(self.data_y)  # number of queries

    #
    def __getitem__(self, idx):
        """
        feature, relevance label, mask, query id of a given index
        """
        x = self.data_x[idx]  # x.shape: (max_nr_docs, fea_dim)
        y = self.data_y[idx]  # y.shape: (max_nr_docs,)
        num_docs = self.data_num_docs[idx]  #
        mask_docs = torch.zeros(self.max_num_docs, dtype=torch.bool)
        mask_docs[:num_docs] = 1
        if self.return_info:
            q_id = self.data_qids[idx]
            sample = {'x': x, 'y': y, 'mask': mask_docs, 'query_id': q_id}
        else:
            sample = {'x': x, 'y': y, 'mask': mask_docs}
        return sample
