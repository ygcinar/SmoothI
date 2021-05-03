"""
some parts are taken from cs230 website
https://github.com/cs230-stanford/cs230-code-examples
"""

import json
import logging
import os
import shutil

import torch
import time
from collections import OrderedDict

from parameters import default_params

PARAMS_ABBRV = OrderedDict(
    {'gamma':'g', 'beta':'b', 'delta':'d', 'alpha':'a', 'learning_rate': 'lr', 'weight_lower_dim': 'd', 'reg_lambda': 'reg', 'batch_size': 'bs', 'num_epochs': 'e',
      'batch_norm': 'bn', 'kernel_size': 'ks', 'dropout_p': 'do', 'num_layers':'nl'})

INV_PARAMS = OrderedDict(
    {'g':'gamma', 'b':'beta', 'd':'delta', 'a':'alpha', 'lr':'learning_rate', 'd':'weight_lower_dim', 'reg':'reg_lambda', 'bs':'batch_size', 'e':'num_epochs',
     'do':'dropout_p', 'nl':'num_layers'})

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, save_last=True, is_first=False, save_each=False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar' if save_last=True. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    # logging.info('Starts to save checkpoint')
    if save_last:
        strt_time = time.time()
        torch.save(state, filepath)
        end_time = time.time()
        time_elapsed = end_time - strt_time
        logging.info('it took {} to save checkpoint'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    if save_each:
        filepath = os.path.join(checkpoint, 'model/epoch{}.pth.tar'.format(str(state['epoch']).zfill(2)))
        if not os.path.exists(os.path.join(checkpoint, 'model')):
            print("Checkpoint Directory does not exist! Making directory {}".format(os.path.join(checkpoint, 'model')))
            os.mkdir(os.path.join(checkpoint, 'model'))
        torch.save(state, filepath)
    if is_best:
        if save_last:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        else:
            strt_time = time.time()
            torch.save(state, os.path.join(checkpoint, 'best.pth.tar'))
            end_time = time.time()
            time_elapsed = end_time - strt_time
            logging.info('it took {} to save checkpoint'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    if is_first:
        strt_time = time.time()
        torch.save(state, os.path.join(checkpoint, 'initial.pth.tar'))
        end_time = time.time()
        time_elapsed = end_time - strt_time
        logging.info('it took {} to save checkpoint'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print('checkpoint: {}'.format(checkpoint))
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint



def save_params_to_json(d, json_path):
    """Saves dict of params in json file

    Args:
        d: (dict) of parameter values
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def args_to_params(args):
    params = default_params(args.model_name, args.data_name, criterion=args.criterion)
    params, updated_args = update_params_w_args(params, args)
    json_path = args.model_dir # updated_args_to_experiment_path(os.path.join(params['par_dir'], args.model_dir), updated_args)
    json_file = os.path.join(json_path, 'params.json')
    try:  # check if the parameter file has already exists
        params = Params(json_file)
    except:
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        save_params_to_json(params, json_file)
        assert os.path.isfile(json_file), "No json configuration file found at {}".format(json_path)
        params = Params(json_file)
    return params, json_path


def update_params_w_args(params, args):
    args_ignore_if_set = ['save_each_epoch', 'dont_continue', 'load_params', 'model_name', 'criterion', 'model_dir', 'restore_file', 'fold', 'nr_folds', 'cv', 'gpu', 'save_first', 'dont_train', 'dont_seed_fold']
    not_nan_args = OrderedDict()
    for (key, value) in vars(args).items():
        if (vars(args)[key] is not None) and (not key in args_ignore_if_set):
            params[key] = value
            not_nan_args[key] = value

    return params, not_nan_args



