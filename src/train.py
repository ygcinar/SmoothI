from dataloader import *
import pickle
import utils
from utils import args_to_params
import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import trange
import argparse

from models import *
from evaluate import evaluate
from losses import ListwiseSmoothIPKLoss, ListwiseSmoothINDCGKLoss

SEED = 12345

import numpy as np
np.random.seed(SEED)

import time


def parse_args():
    parser = argparse.ArgumentParser('one fold of computation of (train-val) of ir ranking')
    parser.add_argument('--model_dir', default=None, help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    parser.add_argument('--fold', default=1, type=int, help='which fold to be run')
    parser.add_argument('--sinit', default=1, type=int, help='which fold to be run')
    parser.add_argument('--nr_folds', default=5, type=int, help='number of cross validation folds')
    parser.add_argument('--cv', action="store_true", default=False,
                        help='if would like to apply cross-validation from fold 1 to nr_folds')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='if True, forces to use gpu')
    parser.add_argument('--save_first', action="store_true", default=False, help='if True, save the initialized state of the model')
    parser.add_argument('--dont_seed_fold', action="store_true", default=False, help='if True, dont multiply global seed with (int) fold')
    parser.add_argument('--dont_continue', action="store_true", default=False, help='to start freshly initialized model and do not continue from last checkpoint')
    #
    parser.add_argument('--load_params', action="store_true", default=False)
    parser.add_argument('--data_name', type=str, default='mq07')
    parser.add_argument('--model_name', type=str, default='mlp')
    #
    parser.add_argument('-c', '--criterion', default='smoothi_pk', type=str)
    parser.add_argument('-k', '--K', default=None, type=int)
    parser.add_argument('-a', '--alpha', default=None, type=float)
    parser.add_argument('-d', '--delta', default=None, type=float)
    #
    parser.add_argument('-l', '--learning_rate', default=None, type=float)
    parser.add_argument('--hidden_sizes', default=None, type=int, nargs='+')
    parser.add_argument('--num_layers', default=None, type=int)
    parser.add_argument('--batch_norm', type=int, choices=[0, 1], default=None,
                        help='Switch to enable to use of batchnorm(1 = batchnorm enabled)')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--num_epochs', default=None, type=int)
    #
    return parser.parse_args()


def inner_loop_data_load_model_call(params, sample_batched, model, device='cpu'):
    """
    Loads minibatch data and calls the model and returns the model output and target values
    Parameters
    ----------
    params: (Params) hyperparameters/arguments
    sample_batched: (dict) minibatches of data
    model: (torch.nn.Module) the neural network
    device: (string) cpu or cuda device

    Returns
    -------
    output_batch: torch tensor of shape (n_minibatch, n_documents) - scores, output of (torch.nn.Module) the neural network
    output_probs_batch: torch tensor of shape (n_minibatch, n_documents) - probabilitie, output of (torch.nn.Module) the neural network
    y: torch tensor of shape (n_minibatch, n_documents) - relevance labels
    """
    # Send the data and label to the device
    x, y, mask = sample_batched['x'].to(params.tensortype).to(device), sample_batched['y'].to(params.tensortype).to(device), sample_batched['mask'].to(device)
    output = model(x, mask)
    # model call
    output_batch, output_probs_batch = output
    return output_batch, output_probs_batch, y


def inner_loop_loss_calculate_and_weight_update(model, model_output, optimizer, loss_fn, params, summ, t, i, running_loss, loss_avg, accu_y, accu_probs, epoch=None):
    """
    Args:
        model: (torch.nn.Module) the neural network
        model_output: (output_batch, output_probs_batch, y) output of inner_loop_data_load_model_call function
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters/arguments
        summ: (list) metric summary
        t: (trange) of tqdm for progress bar
        i: (int) iteration count
        running_loss: (float) running loss
        loss_avg: (float) average loss value
        accu_y: torch tensor of shape (n_minibatch*epoch, 1)
        accu_probs: torch tensor of shape (n_minibatch*epoch, n_documents)
        epoch: (int) epoch count

    """
    output_batch, output_probs_batch, y = model_output
    # calculating the loss between prediction and target
    loss = loss_fn(output_batch, y)
    #
    if torch.isnan(loss):  # NaN loss check!
        logging.info('Loss is NaN!!!')
        assert not torch.isnan(loss)
    # calculating gradients for back propagation
    loss.backward()
    # performs updates using calculated gradients
    optimizer.step()
    running_loss += loss.item()  # accumulating loss
    ###
    # print statistics
    accu_y = torch.cat((accu_y, y), 0)
    accu_probs = torch.cat((accu_probs, output_probs_batch.detach()), 0)
    if i % params.save_summary_steps == params.save_summary_steps - 1:
        loss = running_loss / params.save_summary_steps  # average loss of "params.save_summary_steps" number of iterations
        logging.info('epoch: %d, iteration: %5d loss: %.3f' % (epoch + 1, i + 1, loss))
        running_loss = 0.0
        # batch_scores = Scores(target=accu_y, pred=accu_probs, threshold=0.5)
        # metrics_dict = batch_scores.dict_scores()
        # #
        # summary_batch = {metric: metrics_dict[metric] for metric in metrics_dict.keys()}
        summary_batch = {}
        summary_batch['loss'] = loss
        summ.append(summary_batch)
        #
    # update the average loss
    loss_avg.update(loss)
    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    return model, summ, t, running_loss, loss_avg, accu_y, accu_probs


def train_model(model, optimizer, loss_fn, dataloader, params, epoch=None, device='cpu'):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        epoch: (int) epoch count
        device: (string) cpu or cuda device
    """
    # set model to training mode
    model.train()
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # Use tqdm for progress bar
    num_steps = len(dataloader)
    t = trange(num_steps)
    running_loss = 0.0
    #
    accu_y = torch.tensor([]).to(device)
    accu_probs = torch.tensor([]).to(device)
    for i, sample_batched in zip(t, dataloader):
        # zero the parameter gradients -- clear previous gradients, compute gradients of all variables
        optimizer.zero_grad()
        # forward
        model_output = inner_loop_data_load_model_call(params, sample_batched, model, device=device)
        # backward + optimize
        model, summ, t, running_loss, loss_avg, accu_y, accu_probs = inner_loop_loss_calculate_and_weight_update(
            model, model_output, optimizer, loss_fn, params, summ, t, i, running_loss, loss_avg, accu_y,
            accu_probs, epoch=epoch)
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def main_train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, params, model_dir, restore_file=None, tb_writer=None, device='cpu', save_each_epoch=False, evol_val=True):
    """
    Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validation data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters/arguments
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        tb_writer: (SummaryWriter) tensorboard writer
        device: (string) cpu or cuda device
        save_each_epoch: (bool) save model parameters after each epoch if it's set True
        evol_val: (bool) progress of validation error
    """
    if save_each_epoch:
        utils.save_checkpoint({'epoch': 0,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=False,
                              checkpoint=model_dir, save_last=False, save_each=save_each_epoch)
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint_dict = utils.load_checkpoint(restore_path, model, optimizer)
        epoch_start_ind = checkpoint_dict['epoch']
    else:
        epoch_start_ind = 0
    if params.score_to_select == 'loss':
        best_val_score = np.inf
    else:
        best_val_score = 0.0 # 0.0 9f accuracy is used then it's 0.0 and best value is compared >=
    #
    if evol_val:
        prog_val = []
    for epoch in range(epoch_start_ind, params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics_mean = train_model(model, optimizer, loss_fn, train_data, params, epoch=epoch, device=device)
        # Evaluate for one epoch on validation set
        print('starting to evaluate')
        val_metrics = evaluate(model, loss_fn, val_data, params, device=device)
        #
        val_score = val_metrics[params.score_to_select]
        if params.score_to_select == 'loss':
            is_best = val_score <= best_val_score
        else:
            is_best = val_score >= best_val_score
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir, save_last=True, save_each=save_each_epoch)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_score = val_score
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        if evol_val:
            prog_val.append(val_metrics)
        if tb_writer:
            tb_writer.add_scalars('Loss', {'train':train_metrics_mean['loss'], 'val':val_metrics['loss']}, epoch)
            tb_writer.add_scalars('Val/nDCG', {key: val_metrics[key] for key in ['nDCG1', 'nDCG5', 'nDCG10']}, epoch)
            tb_writer.add_scalars('Val/P', {key: val_metrics[key] for key in ['P1', 'P5', 'P10']}, epoch)
            print('Epoch: {} | Validation loss: {}'.format(epoch, val_metrics['loss']), flush=True)
        print('Validation loss: {}'.format(val_metrics['loss']), flush=True)
    if evol_val:
        pickle.dump(prog_val, open(os.path.join(model_dir, 'val_metrics_s.pkl'), 'wb'))
    logging.info('done training and validation.')


def initialize_model(params, device='cpu'):
    """
    Args:
        params: (Params) hyperparameters/arguments
        device: (string) cpu or cuda device

    Returns:
         model: (torch.nn.Module) the neural network
    """
    if 'mlp' in params.model_name:
        model = MLPRankNet(params, device=device)
    model = model.cuda() if params.cuda else model
    #
    if torch.cuda.device_count() > 1:
        logging.info("It is using {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    #
    return model


def initialize_dataloader(params, fold):
    """
    Args:
        params: (Params) hyperparameters/arguments
        fold: (int) data fold

    Returns:
        train_dataloader: (torch.utils.data.DataLoader) a generator that generates minibatches of train set
        val_dataloader: (torch.utils.data.DataLoader) a generator that generates minibatches of validation set
    """
    #
    if str(fold):
        train_batch = LETORDatasetListwise(params.train_data_file % fold)
        val_batch = LETORDatasetListwise(params.val_data_file % fold, return_info=True)
    else:
        train_batch = LETORDatasetListwise(params.train_data_file)
        val_batch = LETORDatasetListwise(params.val_data_file, return_info=True)
    #
    train_dataloader = DataLoader(train_batch, batch_size=params.batch_size, shuffle=params.shuffle_data, num_workers=params.num_worker)
    val_dataloader = DataLoader(val_batch, batch_size=params.batch_size, shuffle=False, num_workers=params.num_worker)
    return train_dataloader, val_dataloader


def initialize_loss_and_optimizer(params, model, device='cpu'):
    """
    Args:
        params: (Params) hyperparameters/arguments
        model: (torch.nn.Module) the neural network
        device: (string) cpu or cuda device

    Returns:
        criterion: (nn.Module) that takes batch_output and batch_labels and computes the loss for the batch
        criterion: (nn.Module) that takes batch_output and batch_labels and computes the loss for the batch
    """
    if device == "cuda:0":
        params.cuda = True
    if params.criterion == 'listSmoothI_pk_loss':
        criterion = ListwiseSmoothIPKLoss(alpha=params.alpha, delta=params.delta,
                                      stop_grad=params.stop_grad, device=device, K=params.K)
        criterion = criterion.cuda() if params.cuda else criterion
    elif params.criterion == 'listSmoothI_ndcg_loss':
        criterion = ListwiseSmoothINDCGKLoss(alpha=params.alpha, delta=params.delta,
                                         stop_grad=params.stop_grad, device=device, K=params.K, rank_list_length=params.rank_list_length)
        criterion = criterion.cuda() if params.cuda else criterion

    else:
        logging.error('unknown criterion: {}'.format(params.criterion))
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    return criterion, optimizer



def main(args):
    if args.cv:
        folds = range(args.fold, args.nr_folds + 1)
    else:
        folds = [args.fold]
    for fold in folds:
        loop_restore_file = args.restore_file
        if args.load_params:
            json_path = os.path.join(args.model_dir, 'params.json')
            assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
            params = utils.Params(json_path)
            params.tensortype = torch.float32
            args.model_dir_fold = os.path.join(args.model_dir, 'fold%s/' % fold)
        else:
            params, exp_path = args_to_params(args)
            if params.tensortype == 'float32':
                params.tensortype = torch.float32
            args.model_dir_fold = os.path.join(exp_path, 'fold%s/' % fold)
            args.model_dir = exp_path
            if args.cv:
                 args.load_params = True
        # Set the random seed for reproducible experiments
        torch.manual_seed(SEED)
        if params.cuda: torch.cuda.manual_seed_all(SEED)
        #
        if not os.path.exists(args.model_dir_fold):
            os.makedirs(args.model_dir_fold)
        #
        # reset logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Set the logger
        utils.set_logger(os.path.join(args.model_dir_fold, 'train.log'))
        #
        # parent_dir = [folder for folder in args.model_dir_fold.split('/') if 'experiment' in folder][0]
        tb_dir = args.model_dir_fold #args.model_dir_fold.replace(parent_dir, parent_dir + '/tb_logs').replace('/fold', '_fold')
        logging.info('Saving tensorboard logs to {}'.format(tb_dir))
        tb_writer = SummaryWriter(tb_dir)
        #
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('using {}'.format(device))
        if args.gpu:
            assert device != 'cpu'
        # save model parameters before training
        if args.save_first:
            model = initialize_model(params, device=device)
            criterion, optimizer = initialize_loss_and_optimizer(params, model, device=device)
            utils.save_checkpoint({'epoch': 0,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=False,
                                  checkpoint=args.model_dir_fold, save_last=False, is_first=True)
        logging.info("Loading the datasets...")
        #  getting training data in minibatches
        train_dataloader, val_dataloader = initialize_dataloader(params, fold)
        # initialize model torch.nn layer
        model = initialize_model(params, device=device)
        # initialize training criterion and optimizer
        criterion, optimizer = initialize_loss_and_optimizer(params, model, device=device)
        #
        logging.info('parameters: {}'.format(params.__dict__))  # log parameters
        #
        if args.dont_continue:
            loop_restore_file = None
        else:
            restore_path = os.path.join(args.model_dir_fold, 'last.pth.tar')
            if os.path.exists(restore_path):
                logging.info('Restoring from last.pth.tar')
                loop_restore_file = 'last'
        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        main_train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, criterion, params,
                                args.model_dir_fold,
                                loop_restore_file, tb_writer=tb_writer, device=device, evol_val=True)
        logging.info("- done.")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    args = parse_args()
    if not args.dont_seed_fold:
        if str(args.fold):
            SEED = SEED * int(args.fold)
        else:
            SEED = SEED * args.sinit
        np.random.seed(SEED)
        print('SEED after seed*fold', SEED)
    logging.info('args: {}'.format(args))
    main(args)
