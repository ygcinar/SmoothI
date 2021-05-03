"""
default parameters
"""

import torch


def default_params_(params_dict, model_name, criterion=None):
    criterion_dict = {'smoothi_pk':'listSmoothI_pk_loss', 'smoothi_ndcg':'listSmoothI_ndcg_loss'}
    params_dict['criterion'] = criterion_dict[criterion]
    params_dict['pos_fn'] = 'sub'
    params_dict['alpha'] = 1
    params_dict['delta'] = 0.1
    params_dict['stop_grad'] = True
    params_dict['K'] = 10
    params_dict['model_name'] = "{}_{}".format(model_name, params_dict['criterion'])
    rw = model_name
    params_dict['activation_fn'] = 'relu'  # "activation_fn": "relu",
    params_dict['num_layers'] = 1
    params_dict['hidden_sizes'] = [1024] #
    params_dict['num_epochs'] = 50
    params_dict['floatX'] = 'float32'
    params_dict['epsilon'] = 1.1920929e-07  # np.finfo(np.float32).eps
    params_dict['shuffle_data'] = True  # during creating the minibatches shuffle the data if True
    params_dict['batch_size'] = 128
    params_dict['num_worker'] = 0
    params_dict['sampler'] = 'random'
    params_dict['time'] = False
    params_dict['learning_rate'] = 1e-3
    params_dict['batch_norm'] = True
    params_dict['save_summary_steps'] = 1
    params_dict['tensortype'] = 'float32'
    params_dict['cuda'] = torch.cuda.is_available() # use GPU if available
    params_dict['masked_comp'] = False
    params_dict['score_to_select'] = 'loss'
    return params_dict, rw


def default_params(model_name,  dataname='mq07', criterion=None):
    #data
    data_folds_parent_dir = '../data/MQ/MQ2007_pkl/Fold%s/'
    params_dict = {
        'test_data_file': data_folds_parent_dir + 'test.pkl',
        'val_data_file': data_folds_parent_dir + 'vali.pkl',
        'train_data_file': data_folds_parent_dir + 'train.pkl',
        'data_name': dataname}
    if dataname == 'mq07':
        params_dict['rank_list_length'] = 147
        params_dict['fea_dim'] = 46  #
    elif dataname == 'mq08':
        params_dict['rank_list_length'] = 121
        params_dict['fea_dim'] = 46  #
    elif dataname == 'ms30':
        params_dict['rank_list_length'] = 1251
        params_dict['fea_dim'] = 136  #
    elif dataname == 'ylr':
        params_dict['rank_list_length'] = 139
        params_dict['fea_dim'] = 699  #
    # model default hyperparameters (arguments)
    params_dict, rw = default_params_(params_dict, model_name, criterion=criterion)
    return params_dict


