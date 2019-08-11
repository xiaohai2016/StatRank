"""
Neural network training utilities
"""
import copy
import operator
import numpy as np
import torch.optim as optim
import models
import metrics
import data_loader


def run_epoch(batch_iter, full_model, loss_compute, log_interval=50):
  """
  A Generic Training Loop.

  Given a batch data iterator, a model and a loss computation function, runs
  a training loop and report losses (and other information) every log_interval steps.

  Args:
    batch_iter (DataLoader): a torch.utils.data.DataLoader object.
    full_model (nn.Module): a nueral network module.
    loss_compute (functional): a function for loss computing that can adapt to CPU/GPUs.
    log_interval (int): the logging interval.

  Returns:
    loss: loss per sample for the epoch of training.
  """
  total_loss = 0
  samples = 0
  total_samples = 0
  #start = time.time()
  for i, batch in enumerate(batch_iter):
    # BUGBUG: fake batching
    labels = batch[1]
    features = batch[2]
    out = full_model.forward(features)
    loss = loss_compute(out, labels)
    total_loss += loss
    total_samples += 1
    samples += 1
    if i % log_interval == 0:
      #elapsed = time.time() - start
      #print("Epoch Step: %d Loss: %f Samples per Sec: %f" % \
      #        (i, loss / features.size()[0], samples / elapsed))
      #start = time.time()
      samples = 0
  return total_loss / total_samples


def dump_metrics(k_vals, ndcg_ks_list, err_ks_list, column_head='Fold'):
  """Format nDCG and ERR metrics and dump on the stdout"""
  if ndcg_ks_list:
    print("")
    print("".join([f"{column_head},\t"] +
                  ['NDCG@' + str(k) + ',' + (' ' if k < 10 else '') for k in k_vals]))
    for idx, ndcg_ks in enumerate(ndcg_ks_list):
      print(",\t".join([str(idx + 1)] + ["{:.4f}".format(score) for score in ndcg_ks]))
    np_ndcg_ks_list = np.asarray(ndcg_ks_list, dtype=np.float64)
    ndcg_ks_average = np.mean(np_ndcg_ks_list, axis=0)
    ndcg_ks_std = np.std(np_ndcg_ks_list, axis=0)
    print(",\t".join(['AVG'] + ["{:.4f}".format(score) for score in ndcg_ks_average.tolist()]))
    print(",\t".join(['STD'] + ["{:.4f}".format(score) for score in ndcg_ks_std.tolist()]))

  if err_ks_list:
    print("")
    print(",\t".join([f"{column_head}"] + ['ERR@' + str(k) for k in k_vals]))
    for idx, err_ks in enumerate(err_ks_list):
      print(",\t".join([str(idx + 1)] + ["{:.4f}".format(score) for score in err_ks]))
    np_err_ks_list = np.asarray(err_ks_list, dtype=np.float64)
    err_ks_average = np.mean(np_err_ks_list, axis=0)
    err_ks_std = np.std(np_err_ks_list, axis=0)
    print(",\t".join(['AVG'] + ["{:.4f}".format(score) for score in err_ks_average.tolist()]))
    print(",\t".join(['STD'] + ["{:.4f}".format(score) for score in err_ks_std.tolist()]))

  return k_vals, ndcg_ks_average, err_ks_average

class SimpleLossCompute:
  """
  A simple loss compute function that drives training.

  Attributes:
    criterion: the loss function.
    optimizer: the optimizer object for training.
  """
  def __init__(self, criterion, optimizer=None):
    self.criterion = criterion
    self.optimizer = optimizer

  def __call__(self, out, label):
    loss = self.criterion(out.contiguous().view(-1, out.size(-1)),
                          label.contiguous().view(-1, label.size(-1)))
    if self.optimizer is not None:
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    return loss.data.item()

class BaseTrainer(object):
  """
  The base class for a trainer object.

  Supports running of training and evluations.

  Attributes:
    name (str): the name of a trainer reflecting the choise of modeling and criterion.
    train_iter: the training data batch iterator constructed from torch.utils.data.DataLoader.
    valid_iter: the validation data batch iterator constructed from torch.utils.data.DataLoader.
    test_iter: the testing data batch iterator constructed from torch.utils.data.DataLoader.
    full_model: the full model for document scoring.
    criterion: the loss function.
    optimizer: the optimizer object for training.
    loss_compute: the loss computation function. We are using the simple CPU based
                  loss compute function for now. This can be replaced with a class
                  with parallel loss computation across multiple GPUs.
  """

  def __init__(self, name=None,
               train_iter=None, valid_iter=None, test_iter=None,
               full_model=None, criterion=None,
               optimizer=None):
    """Set up modules for training."""
    self.name = name
    self.train_iter = train_iter
    self.valid_iter = valid_iter
    self.test_iter = test_iter
    self.full_model = full_model
    self.criterion = criterion
    self.optimizer = optimizer
    self.loss_compute = SimpleLossCompute(criterion=criterion, optimizer=optimizer)

  def run_training(self, epoch_count, k_vals, log_interval=50, max_grade=2):
    """Run training for epoch_count epochs."""
    ndcg_cutt_off = 10
    best_ndcg = 0.0
    best_epoch = -1
    best_model = None
    for epoch in range(epoch_count):
      self.full_model.train()
      loss = run_epoch(
        self.train_iter, self.full_model, self.loss_compute, log_interval=log_interval)
      self.full_model.eval()
      ndcg_ks, _ = self.run_evaluation(self.full_model, self.valid_iter,
                                       k_vals=[ndcg_cutt_off], max_grade=max_grade)
      ndcg_k = ndcg_ks[0]
      if ndcg_k > best_ndcg:
        best_model = copy.deepcopy(self.full_model)
        best_epoch = epoch
        best_ndcg = ndcg_k
        print(f"\tEpoch: {best_epoch}, training loss: {loss}, NDCG@{ndcg_cutt_off}: {best_ndcg}")
      else:
        print(f"\t\tEpoch: {epoch}, training loss: {loss}, NDCG@{ndcg_cutt_off}: {ndcg_k}")
    best_model.eval()
    ndcg_ks, err_ks = self.run_evaluation(best_model, self.test_iter,
                                          k_vals=k_vals, max_grade=max_grade)
    print(f"\nTest scores NDCG:{ndcg_ks}; ERR:{err_ks} with best epoch: {best_epoch}")
    return ndcg_ks, err_ks

  def run_evaluation(self, model, data_iter, k_vals=None, max_grade=2):
    """Run inference with the model and compute performance metrics."""
    if k_vals is None:
      k_vals = []
    sum_ndcg_at_k = [0.0 for _ in range(len(k_vals))]
    sum_err_at_k = [0.0 for _ in range(len(k_vals))]
    sample_count = 0
    for _, batch in enumerate(data_iter):
      labels = batch[1]
      features = batch[2]
      for lbls, scrs in zip(labels.numpy(), model.forward(features).detach().numpy()):
        sum_ndcg_at_k = list(map(operator.add, sum_ndcg_at_k,
                                 [metrics.ndcg_score(lbls, scrs, k=k) for k in k_vals]))
        sum_err_at_k = list(map(operator.add, sum_err_at_k,
                                [metrics.err(lbls, scrs, k=k,
                                             max_grade=max_grade) for k in k_vals]))
        sample_count += 1
    return [x / sample_count for x in sum_ndcg_at_k], [x / sample_count for x in sum_err_at_k]


class MQ200XTrainer(BaseTrainer):
  """
  Implementation of the ListNet training.

  Attributes:
    use_mq2007 (bool): to train on MQ2007 dataset (false means MQ2008).
    feature_count (int): the number of features.
    lrate (float): the learning rate.
    weight_decay (float): the weight decay coefficient.
  """

  def __init__(self, use_mq2007=True, feature_count=46, lrate=1e-3,
               weight_decay=1e-3, model='simple_one_layer'):
    """Construct ListNet training modules."""
    super(MQ200XTrainer, self).__init__(name="ListNet")
    self.use_mq2007 = use_mq2007
    self.feature_count = feature_count
    self.lrate = lrate
    self.weight_decay = weight_decay
    self.model = model

  def train(self, criterion, epoch_count=200, log_interval=50):
    """Train on MQ2007 dataset with ListNet algorithm"""
    k_vals = [1, 3, 5, 10, 20, 50]
    ndcg_ks_list = []
    err_ks_list = []
    for fold in range(1, 6):
      print(f"Fold {fold}")

      full_model = None
      if self.model == "simple_one_layer":
        full_model = models.SimpleOneLayerLinear(self.feature_count)
      else:
        full_model = models.SimpleThreeLayerLinear(self.feature_count)
      optimizer = optim.Adam(
        full_model.parameters(),
        lr=self.lrate, weight_decay=self.weight_decay)

      path = 'resources/' + ('MQ2007' if self.use_mq2007 else 'MQ2008') + \
            '/Fold' +str(fold) + '/'
      train_iter = data_loader.get_ms_dataset(path + 'train.txt')
      valid_iter = data_loader.get_ms_dataset(path + 'vali.txt', shuffle=False)
      test_iter = data_loader.get_ms_dataset(path + 'test.txt', shuffle=False)
      super(MQ200XTrainer, self).__init__(
        name="ListNet",
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        full_model=full_model,
        criterion=criterion,
        optimizer=optimizer
        )
      ndcg_ks, err_ks = super(MQ200XTrainer, self).run_training(
        epoch_count, k_vals=k_vals,
        log_interval=log_interval)
      ndcg_ks_list.append(ndcg_ks)
      err_ks_list.append(err_ks)
    return dump_metrics(k_vals, ndcg_ks_list, err_ks_list)
