"""
Definitions of different criteria for loss computation.
"""
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
