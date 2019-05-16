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

class CDFNormalN(torch.autograd.Function):
  '''
  The PyTorch OP corresponding to Cumulative Distribution functin of a normal
  distribution. We used a simple approximation for the forward pass.
  http://www.hrpub.org/download/20140305/MS7-13401470.pdf
  '''
  @staticmethod
  def forward(ctx, *args, **kwargs):
    '''
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    '''
    n_sample = args[1].double()
    inx = args[0]
    inx_square = inx * inx
    indi = (inx < 0).double()
    inx = torch.abs(inx)
    normal_val = np.sqrt(0.5 / np.pi) * torch.exp(-inx_square / 2.0)
    cdf_out = 1.0 - normal_val / (0.226 + 0.64 * inx + 0.33 * torch.sqrt(inx_square + 3))

    ctx.save_for_backward(n_sample * normal_val, n_sample)
    return n_sample * torch.abs(cdf_out - indi)

  @staticmethod
  def backward(ctx, *grad_outputs):
    '''
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    '''
    grad_output = grad_outputs[0]
    fd_output, n_sample = ctx.saved_tensors
    #chain rule
    bk_output = grad_output * fd_output

    return bk_output, torch.zeros_like(n_sample)

class StatRankCriterion(nn.Module):
  """
  Implementation of the StatRank algorithm
"""

  def forward(self, *inputs):
    """
    The main forward method.
    """
    predictions = inputs[0]
    labels = inputs[1]
    # BUGBUG, for batch size larger than 1, we need to passing
    # in a mask based on the input data to compute samples
    mask = torch.zeros(labels.shape, dtype=torch.int32) + 1.0
    n_samples = torch.sum(mask, dim=-1)

    prediction_mean = torch.mean(predictions, dim=-1, keepdim=True)
    predition_var = torch.var(predictions, dim=-1, keepdim=True)
    prediction_normalized = (predictions - prediction_mean) / torch.sqrt(predition_var)
    predicted_pos = CDFNormalN.apply(prediction_normalized, n_samples)
    rel_val = torch.pow(2.0, labels) - 1.0
    return torch.sum(rel_val / torch.log2(1.0 + predicted_pos), dim=-1)

class NDCGFunctional(torch.autograd.Function):
  '''
  The PyTorch OP corresponding to nDCG calculation and its back propogation
  subgradients.
  '''
  @staticmethod
  def forward(ctx, *args, **kwargs):
    '''
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    '''
    prediction_scores = args[0]
    relevance_lables = args[1]
    k_val = args[2]

    _, score_sorted_index = torch.sort(prediction_scores, descending=True)
    sorted_labels, _ = torch.sort(relevance_lables, descending=True)
    relevance_with_predictions = torch.gather(relevance_lables, -1, score_sorted_index)

    denoms = torch.log2(torch.arange(k_val, dtype=torch.float) + 2.0)   #discounting factor
    denoms = torch.unsqueeze(denoms.double(), dim=0)
    dcg = torch.sum(relevance_with_predictions[:, :k_val] / denoms, dim=-1)
    idcg = torch.sum(sorted_labels[:, :k_val] / denoms, dim=-1)

    labels_m_c = torch.unsqueeze(relevance_lables, dim=-1)
    labels_m_r = torch.unsqueeze(relevance_lables, dim=-2)
    lables_m = labels_m_c - labels_m_r

    scores_m_c = torch.unsqueeze(prediction_scores, dim=-1)
    scores_m_r = torch.unsqueeze(prediction_scores, dim=-2)
    scores_m = scores_m_c - scores_m_r

    signal_m = torch.min(scores_m * lables_m, torch.zeros_like(scores_m))
    lables_m_sign = (lables_m < 0).double() - (lables_m > 0).double()
    signal_final = torch.sum(lables_m_sign * signal_m, dim=-1)

    ctx.save_for_backward(signal_final, relevance_lables)
    return torch.sum(1.0 - dcg/idcg)

  @staticmethod
  def backward(ctx, *grad_outputs):
    '''
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    '''
    grad_output = grad_outputs[0]
    signal_final, arg2 = ctx.saved_tensors
    #chain rule
    bk_output = grad_output * signal_final

    return bk_output, torch.zeros_like(arg2), None

class CrossRankCriterion(nn.Module):
  """
  Implementation of the CrossRank algorithm
"""

  def forward(self, *inputs):
    """
    The main forward method.
    """
    predictions = inputs[0]
    labels = inputs[1]

    rel_val = torch.pow(2.0, labels) - 1.0
    return NDCGFunctional.apply(predictions, rel_val, 10)
