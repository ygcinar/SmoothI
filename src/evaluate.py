"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
from models import *
from dataloader import *
from torch.utils.data import DataLoader

import time

import pytrec_eval

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='../experiments/', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--fold', default=1, help='which fold to be run')
parser.add_argument('--gpu', action="store_true", default=False,
                    help='if would like to do centroid calculation using torch-gpu or not')

SEED = 12345
INF = np.inf


def evaluate_inner_loop(model, params, sample_batched, device='cpu'):
    """ Evaluate the model on a sample minibatch, calls the model and returns the model output and target values
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters/arguments
        sample_batched: (dict) minibatches of data
        device: (string) cpu or cuda device

    Returns:
        output_batch: torch tensor of shape (n_minibatch, n_documents) - scores, output of (torch.nn.Module) the neural network
        output_probs_batch: torch tensor of shape (n_minibatch, n_documents) - probabilitie, output of (torch.nn.Module) the neural network
        y: torch tensor of shape (n_minibatch, n_documents) - relevance labels
    """
    # cast to tensor type and change tensor if cuda device
    x, y, mask = sample_batched['x'].to(params.tensortype).to(device), sample_batched['y'].to(params.tensortype).to(
        device), sample_batched['mask'].to(device)
    # model call
    output = model(x, mask)
    output_batch, output_probs_batch = output  # model output: scores, probabilities
    return (output_batch, output_probs_batch), y


def rank_scores(scores, y, return_item_scores=False):
    """
    Args:
        scores: ndarray of shape (n_queries, n_documents) - scores of the documents for a query q which are predicted by the model
        y: ndarray of shape (n_queries, n_documents) - relevance labels
        return_item_scores: (bool) if return_instance_scores is True, then evaluate_by_iterating_minibatch returns query-level ranking scores

    Returns:
        perf_eval_scores: (dict) average ranking scores over queries
        instance_scores_dict: (dict) query-level ranking scores
    """
    strt_time = time.time()
    bs, lly = y.shape
    bs, lls = scores.shape
    ll = min(lly, lls)  # ignore extra padding
    # prepare query relevance and query score dictionaries for pytrec_eval
    qrel_dict = {}
    qscore_dict = {}
    for idx in range(bs):
        qrel_dict['q{}'.format(idx)] = {'d{}'.format(d_idx): int(y[idx, d_idx]) for d_idx in range(ll)}
        qscore_dict['q{}'.format(idx)] = {'d{}'.format(d_idx): float(scores[idx, d_idx]) for d_idx in range(ll)}
    #
    metrics = ['ndcg_cut.1', 'ndcg_cut.5', 'ndcg_cut.10', 'ndcg', 'P.1', 'P.5', 'P.10']
    metrics_name = {'ndcg_cut.1': 'nDCG1', 'ndcg_cut.5': 'nDCG5', 'ndcg_cut.10': 'nDCG10', 'ndcg': 'nDCG', 'P.1': 'P1',
                    'P.5': 'P5', 'P.10': 'P10'}
    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, metrics)  # pytrec_eval relevance evaluator
    res_metrics_dict = evaluator.evaluate(qscore_dict)  # ranking metrics
    # ranking performance per query
    instance_scores_dict = {}
    for m_i in metrics:
        mi = metrics_name[m_i]
        instance_scores_dict[mi] = []
        for q_idx in range(bs):
            instance_scores_dict[mi].append(res_metrics_dict['q{}'.format(q_idx)][m_i.replace('.', '_')])
    #
    # average ranking scores over queries
    perf_eval_scores = {m: np.average(lm) for (m, lm) in instance_scores_dict.items()}
    end_time = time.time()
    time_elapsed = end_time - strt_time
    # time statistics
    logging.info('time elapsed: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    logging.info('len(instance_scores_dict[mi]): {}'.format(len(instance_scores_dict[mi])))
    assert len(instance_scores_dict[mi]) == len(y)  # check scores for each query
    if return_item_scores:
        return perf_eval_scores, instance_scores_dict
    else:
        return perf_eval_scores


def evaluate(model, loss_fn, data_iterator, params, device='cpu'):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates minibatches of data and labels
        params: (Params) hyperparameters/arguments
        tb_writer: (SummaryWriter) tensorboard writer
        device: (string) cpu or cuda device
    Returns:
        metrics_mean: (dict) average metrics
    """
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():  # we dont want to accumulate gradient
        # summary for current eval loop
        summ = []
        # scores for current eval loop
        scores_mtx = None
        # labels for current eval loop
        y_mtx = None
        # compute metrics over the dataset
        init = True
        for i, sample_batched in enumerate(data_iterator):
            # compute scores and probabilities for the current sample
            (output_batch, output_probs_batch), y = evaluate_inner_loop(model, params, sample_batched, device=device)
            loss = loss_fn(output_batch, y)
            # accumulate for session based evaluation
            if not init:
                scores_mtx = np.vstack((scores_mtx, output_batch.cpu().numpy()))  #
                y_mtx = np.vstack((y_mtx, y.cpu().numpy()))  #
            else:
                scores_mtx = output_batch.cpu().numpy()
                y_mtx = y.cpu().numpy()
                init = False
            summary_batch = {'loss':loss.item()}
            summ.append(summary_batch)
        # compute rank scores
        rank_metrics_mean = rank_scores(scores_mtx, y_mtx)
        # compute mean of all metrics in summary
        metrics_mean1 = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
        #
        print(metrics_mean1.keys())
        metrics_mean = {**rank_metrics_mean, **metrics_mean1}
        print(metrics_mean.keys())
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def evaluate_by_iterating_minibatch(model, params, test_dataloader, device='cpu', return_instance_scores=False):
    """
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters/arguments
        test_dataloader: (generator) a generator that generates minibatches of data and labels
        device: (string) cpu or cuda device
        return_instance_scores: (bool) if return_instance_scores is True, then evaluate_by_iterating_minibatch returns query-level ranking scores

    Returns:
        mean_scores: (dict) average ranking scores over queries
        query_level_scores: (dict) query-level ranking scores
        query_ids_accu: ndarray of shape (n_queries, 1)
    """
    # initilize numpy matrices to accumulate scores, query ids, and labels of minibatches
    query_ids_accu = np.empty((0, 1))
    scores_accu = np.empty((0, params.rank_list_length))
    y_accu = np.empty((0, params.rank_list_length))
    with torch.no_grad():  # we dont want to accumulate gradient
        for i, sample_batched in enumerate(test_dataloader):  # loop over minibatches
            # compute scores and probabilities for the current sample
            (output_batch, output_probs_batch), y = evaluate_inner_loop(model, params, sample_batched, device=device)
            # accumulate minibatch scores, query ids, and labels
            output_scores = np.hstack((output_batch.cpu().numpy(), np.zeros(
                (output_batch.shape[0], params.rank_list_length - output_batch.shape[1]))))
            scores_accu = np.vstack((scores_accu, output_scores))
            query_ids_accu = np.vstack((query_ids_accu, np.expand_dims(np.array(sample_batched['query_id']), axis=1)))
            label = np.hstack((y.cpu().numpy(), np.zeros((y.shape[0], params.rank_list_length - y.shape[1]))))
            y_accu = np.vstack((y_accu, label))
    #
    assert len(query_ids_accu) == len(scores_accu)
    assert len(query_ids_accu) == len(y_accu)
    # calculate ranking metrics for the predicted scores
    mean_scores, query_level_scores = rank_scores(scores_accu, y_accu, return_item_scores=True)
    if return_instance_scores:
        return mean_scores, query_level_scores, query_ids_accu
    else:
        return mean_scores


def initialize_model(params, device='cpu'):
    """
        Parameters
        ----------
        params -- (hyper)parameters of the current model

        Returns
        -------
        model -- initialized model according to the model specified in params
    """
    model = MLPRankNet(params, device=device)
    model = model.cuda() if params.cuda else model
    # if there are more than one gpu's make data parallelization
    if torch.cuda.device_count() > 1:
        logging.info("It is using", torch.cuda.device_count(),
                     "GPUs!")  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    #
    return model


def initialize_dataloader(params, fold):
    """
    Args:
        params: (hyper)parameters of the data, model
        fold: which data fold

    Returns:
        test_set
    """
    if '%s' in params.test_data_file:
        test_batch = LETORDatasetListwise(params.test_data_file % fold, return_info=True)
    else:
        test_batch = LETORDatasetListwise(params.test_data_file, return_info=True)
    return test_batch


def evaluate_main(args_model_dir, fold=1, args_gpu=True, args_restore_file='best', return_instance_score=False,
                  set_logger=True):
    """
    Args:
        args_model_dir: (string) directory containing config
        fold: (int) data fold (e.g. 1)
        args_gpu: (bool) if it's True forces to use gpu device
        args_restore_file: (string) name of file to restore from (without its extension .pth.tar)
        return_instance_score: boolean if it's True function returns the instance scores e.g. NDCG per query
        set_logger: boolean if it's true it setups the logger
    Returns:
        instance_scores - if return_instance_score set True, then it returns instance scores e.g. P@1,5,10, NDCG@1,5,10 per query
    """
    # load model hyperparameters/arguments
    json_path = os.path.join(args_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    args_model_dir = os.path.join(args_model_dir, 'fold%s/' % fold)
    params.tensortype = torch.float32
    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('using {}'.format(device))
    # Set the random seed for reproducible experiments
    torch.manual_seed(SEED)
    if params.cuda: torch.cuda.manual_seed(SEED)
    #
    if set_logger:
        # reset logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Get the logger
        utils.set_logger(os.path.join(args_model_dir, 'evaluate.log'))
        print('log path: {}'.format(os.path.join(args_model_dir, 'evaluate.log')))
        # Create the input data pipeline
        logging.info("Creating the dataset...")
    # load data
    if args_gpu:
        assert device != 'cpu'
    # Define the model and dataset
    test_batch = initialize_dataloader(params, fold)
    # getting training data in minibatches
    test_dataloader = DataLoader(test_batch, batch_size=params.batch_size, shuffle=False, num_workers=params.num_worker)
    model = initialize_model(params, device=device)
    # Reload weights from the saved model parameters file
    utils.load_checkpoint(os.path.join(args_model_dir, args_restore_file + '.pth.tar'), model)
    model = model.cuda() if params.cuda else model
    model.eval()  # we do not want to accumulate gradient
    # evaluate the model on the test set
    logging.info("Starting evaluation for the params: {}".format(params.__dict__))
    results = evaluate_by_iterating_minibatch(model, params, test_dataloader, device=device,
                                              return_instance_scores=return_instance_score)
    logging.info('results: {}'.format(results))
    #
    if return_instance_score:
        return results[1]
    else:
        f_res = open(args_model_dir + 'rank_results_fold%s.csv' % fold, 'a')
        f_res.write(','.join(list(results.keys())) + '\n')
        f_res.write(','.join(map(str, list(results.values()))) + '\n')
        save_path = os.path.join(args_model_dir, "metrics_test_{}.json".format(args_restore_file))
        utils.save_dict_to_json(results, save_path)
        logging.info("- done.")


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    logging.info('args: {}'.format(args))
    evaluate_main(args.model_dir, fold=args.fold, args_gpu=args.gpu, args_restore_file=args.restore_file, return_instance_score=False)
