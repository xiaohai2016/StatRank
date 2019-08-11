"""
Definition(s) of inference models for document scores.
"""
import torch
import torch.nn as nn


class SimpleOneLayerLinear(nn.Module):
  """
  Simple one layer feedforward network with sigmoid activation and
  dropout rate of 0.01

  Attributes:
    activation: the activation function.
    dropout: the dropout module.
    net: the neural network.
  """

  def __init__(self, in_features, out_features=1, activation='sigmoid'):
    """
    Initializes the module.

    Parameters:
      in_features (int): the input feature size.
      out_features (int): the output feature size.
      activation (str): the name of the activation function.

    """
    super(SimpleOneLayerLinear, self).__init__()
    self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()
    self.net = nn.Linear(in_features, out_features).double()
    torch.nn.init.xavier_uniform_(self.net.weight)

  def forward(self, *inputs):
    """The forward processing"""
    return torch.squeeze(self.activation(self.net(inputs[0])), dim=-1)


class SimpleThreeLayerLinear(nn.Module):
  """
  Simple three layer feedforward network and dropout rate of 0.01

  Attributes:
    ffnn: the feed-forward neural network.
  """

  def __init__(self, in_features, out_features=1, hidden_dim=100, dropout=0.01,
               hidden_activation='relu', final_activation='sigmoid'):
    """
    Initializes the module.

    Parameters:
      in_features (int): the input feature size.
      out_features (int): the output feature size.
      hidden_dim (int): the hidden layer dimension.
      dropout (float): the dropout rate for the output of the first layer.
      hidden_activation (str): the name of the activation function for the hidden layoers.
      final_activation (str): the name of the activation function for the final layoer.
    """
    super(SimpleThreeLayerLinear, self).__init__()
    head_activation_fn1 = nn.Sigmoid() if hidden_activation == 'sigmoid' else nn.ReLU()
    hidden_activation_fn2 = nn.Sigmoid() if hidden_activation == 'sigmoid' else nn.ReLU()
    final_activation_fn = nn.Sigmoid() if final_activation == 'sigmoid' else nn.ReLU()
    dropout_fn = nn.Dropout(p=dropout)
    self.ffnn = nn.Sequential()

    linear_layer = nn.Linear(in_features, hidden_dim).double()
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    self.ffnn.add_module("linear_layer_1", linear_layer)
    self.ffnn.add_module("act_1", head_activation_fn1)
    self.ffnn.add_module("drop_1", dropout_fn)

    linear_layer = nn.Linear(hidden_dim, hidden_dim).double()
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    self.ffnn.add_module("linear_layer_2", linear_layer)
    self.ffnn.add_module("act_2", hidden_activation_fn2)

    linear_layer = nn.Linear(hidden_dim, out_features).double()
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    self.ffnn.add_module("linear_layer_3", linear_layer)
    self.ffnn.add_module("act_3", final_activation_fn)

  def forward(self, *inputs):
    """The forward processing"""
    return torch.squeeze(self.ffnn(*inputs), dim=-1)
