"""
Definitions of different criteria for loss computation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetTopOneCriterion(nn.Module):
  """
  Implementation of the ListNet algorithm:
  Learning to Rank: From Pairwise Approach to Listwise Approach.
  In Proceedings of the 24th ICML. 129–136.
  """

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    out = inputs[0]
    labels = inputs[1]
    return torch.sum(-torch.sum(F.log_softmax(out, dim=1) * F.softmax(labels, dim=1), dim=1))

class KLDivergenceTopOneCriterion(nn.Module):
  """
  Implementation of the KL divergence algorithm for top 1 probabilities
  """

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    out = inputs[0]
    labels = inputs[1]
    return torch.sum(
            torch.sum(F.softmax(out, dim=1) *
                      (F.log_softmax(out, dim=1) - F.log_softmax(labels, dim=1)), dim=1)
            )

class AlphaDivergenceTopOneCriterion(nn.Module):
  """
  Implementation of the Alpha divergence algorithm for top 1 probabilities
  Formula: D_alpha(out || label)
  """

  def __init__(self, alpha=0.0):
    super(AlphaDivergenceTopOneCriterion, self).__init__()
    threshold = 1.0e-5
    self.alpha = alpha
    self.is_kl_divergence = True if (abs(alpha - 1.0) < threshold) else False
    self.is_cross_entropy = True if abs(alpha) < threshold else False

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    out = inputs[0]
    labels = inputs[1]
    if self.is_kl_divergence:
      return torch.sum(
        torch.sum(F.softmax(out, dim=1) *
                  (F.log_softmax(out, dim=1) - F.log_softmax(labels, dim=1)), dim=1))
    elif self.is_cross_entropy:
      return torch.sum(-torch.sum(F.log_softmax(out, dim=1) * F.softmax(labels, dim=1), dim=1))
    else:
      p = F.softmax(out, dim=1) # pylint: disable=invalid-name
      q = F.softmax(labels, dim=1) # pylint: disable=invalid-name
      alpha = self.alpha
      alpha_c = 1.0 - alpha
      val = alpha * p + alpha_c * q - torch.pow(p, alpha) * torch.pow(q, alpha_c)
      val = val / alpha / alpha_c
      return torch.sum(torch.sum(val, dim=1))

class AlphaDivergenceTopOneAndEntropyRegularizationCriterion(nn.Module):
  """
  Implementation of the Alpha divergence algorithm for top 1 probabilities
  along with entropy regularization. The formula is
  D_alpha(out || label) - lambda * Entropy(out)
  """

  def __init__(self, alpha=0.0, lambd=0.0):
    super(AlphaDivergenceTopOneAndEntropyRegularizationCriterion, self).__init__()
    threshold = 1.0e-5
    self.alpha = alpha
    self.lambd = lambd
    self.is_kl_divergence = True if (abs(alpha - 1.0) < threshold) else False
    self.is_cross_entropy = True if abs(alpha) < threshold else False

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    out = inputs[0]
    labels = inputs[1]
    out_softmax = F.softmax(out, dim=1)
    out_log_softmax = F.log_softmax(out, dim=1)
    if self.is_kl_divergence:
      return torch.sum(
        torch.sum(out_softmax *
                  (out_log_softmax * (1 + self.lambd) - F.log_softmax(labels, dim=1)), dim=1))
    elif self.is_cross_entropy:
      return torch.sum(-torch.sum(out_log_softmax * (F.softmax(labels, dim=1) - out_softmax * self.lambd), dim=1))
    else:
      p = F.softmax(out, dim=1) # pylint: disable=invalid-name
      q = F.softmax(labels, dim=1) # pylint: disable=invalid-name
      alpha = self.alpha
      alpha_c = 1.0 - alpha
      val = alpha * p + alpha_c * q - torch.pow(p, alpha) * torch.pow(q, alpha_c)
      val = val / alpha / alpha_c
      return torch.sum(torch.sum(val + out_log_softmax * out_softmax * self.lambd, dim=1))

class WeightedKLDivergenceTopOneCriterion(nn.Module):
  """
  Implementation of the KL divergence algorithm for top 1 probabilities with entropy regularization
  Formula: KL(out || label) - lambda * Entropy(out)
  Note the KL(out || label) contains (-1.0) * Entropy(out). When searching for the hyper parameter
  lambd, one might want to start from -1.0 instead of from 0.0.
  """

  def __init__(self, lambd=0.0):
    super(WeightedKLDivergenceTopOneCriterion, self).__init__()
    self.lambd = lambd

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    out = inputs[0]
    labels = inputs[1]
    return torch.sum(
      torch.sum(F.softmax(out, dim=1) *
                (F.log_softmax(out, dim=1) * (1 + self.lambd) - F.log_softmax(labels, dim=1)), dim=1))

class LogCumsumExp(torch.autograd.Function):
  '''
  The PyTorch OP corresponding to the operation: log{ |sum_k^m{ exp{pred_k} } }
  '''
  @staticmethod
  def forward(ctx, *args, **kwargs):
    '''
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    '''
    inx = args[0]
    inmax, _ = torch.max(inx, dim=-1, keepdim=True)
    inx_scaled = inx - inmax
    inx_exp = torch.exp(inx_scaled)
    cumsum_tail_2_head = torch.flip(torch.cumsum(torch.flip(inx_exp, dims=[-1]), dim=-1), dims=[-1])
    fd_output = torch.log(cumsum_tail_2_head) + inmax
    ctx.save_for_backward(inx, fd_output)
    return fd_output

  @staticmethod
  def backward(ctx, *grad_outputs):
    '''
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    '''
    grad_output = grad_outputs[0]
    inx, fd_output = ctx.saved_tensors
    #chain rule
    bk_output = grad_output * (torch.exp(inx) * torch.cumsum(torch.exp(-fd_output), dim=-1))

    return bk_output

class ListMLECriterion(nn.Module):
  """
  Implementation of the ListMLE algorithm:
  Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008.
  Listwise Approach to Learning to Rank: Theory and Algorithm.
	In Proceedings of the 25th ICML. 1192–1199.
  """

  def forward(self, *inputs):
    """
    The main forward method.
    We here use the Top-1 approximated ListNet loss,
    which reduces to a softmax and simple cross entropy.
    """
    predictions = inputs[0]
    labels = inputs[1]
    order = torch.argsort(labels, descending=True)
    predictions_sorted = torch.gather(predictions, -1, order)
    final = LogCumsumExp.apply(predictions_sorted)
    return torch.sum(final - predictions_sorted)
