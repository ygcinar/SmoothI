import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed=seed)


class MLPRankNet(nn.Module):
    """
    Feed-forward Neural Network Layer to predict relevance of query-document pairs
    """
    def __init__(self, params, device='cpu'):
        super(MLPRankNet, self).__init__()
        self.rank_list_length = params.rank_list_length
        self.device = device
        self.batch_norm = False
        num_layers = len(params.hidden_sizes)
        hidden_sizes = [params.fea_dim] + params.hidden_sizes
        self.layers = nn.ModuleList([nn.Linear(hidden_sizes[idx], hidden_sizes[idx + 1]) for idx in range(num_layers)])
        #
        self.f_out = nn.Linear(hidden_sizes[-1], 1)
        if params.activation_fn == 'relu':
            self.a_fn = nn.ReLU(inplace=True)
        if params.batch_norm:
            self.batch_norm = True
            self.bn_x = nn.BatchNorm1d(params.fea_dim)
            self.layers_bn = nn.ModuleList([nn.BatchNorm1d(params.hidden_sizes[idx]) for idx in
                                            range(num_layers)])  # applying the batch norm on the input layer as well
        self.masked_comp = params.masked_comp
        self.initialize_weights()
    #
    def initialize_weights(self):
        """
        Initialize nn.Linear layer and batchnorm layer weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    #
    def forward(self, input, mask):
        """
        Args:
            input:  torch tensor of shape (batch_size, num_documents, feature_dimension)
            mask: torch tensor of shape (batch_size, num_documents, feature_dimension)
        Returns:
            out: torch tensor of shape (batch_size, num_documents) - scores: last layer output
            probs: torch tensor of shape (batch_size, num_documents) - probabilities: softmax over last layer output
        """
        bs, n_docs, fea_dim = input.shape
        x = input.view(-1, fea_dim)  # reshape input x.shape: (batch_size*n_docs, fea_dim)
        if self.masked_comp:
            mask = mask.view(-1, )
            x = x[mask]
        if self.batch_norm:
            x = self.bn_x(x)  # apply batch normalization to input layer  # x_1.shape: (batch_size*n_docs, fea_dim)
        # apply feed-forward layer transformation to input x (news features)
        for i, (layer_module, layer_bn_module) in enumerate(zip(self.layers, self.layers_bn)):
            x = layer_module(x)  # x_1.shape: (batch_size*n_docs, hidden_size)
            x = self.a_fn(x)  # apply activation function # x.shape: (batch_size*n_docs, hidden_size)
            if self.batch_norm:
                x = layer_bn_module(
                    x)  # apply batch normalization to hidden layer  # x_1.shape: (batch_size*n_docs, fea_dim)
        x2 = self.f_out(x)
        if self.masked_comp:
            out = torch.zeros((bs * n_docs, 1), device=self.device)
            out[mask] = x2
            out = out.view(bs, n_docs)
        else:
            out = x2.view(bs, n_docs)
        # layer - activation - bn - dropout
        probs = F.softmax(out, dim=1)
        return out, probs  # .shape: (batch_size, n_docs)
