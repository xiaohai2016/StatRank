"""
The main entry point - handles command line options and invocations
"""
import argparse
import numpy as np
import torch
import training
import criteria


def parse_args():
  """To parse command line arguements"""
  parser = argparse.ArgumentParser('StatRank version 1.0.')
  parser.add_argument('--repeat',
                      help='the number of repeating runs to collect the metrics average',
                      type=int, default=10)
  parser.add_argument('--data-set',
                      help='the dataset to train on',
                      choices=('mq2007', 'mq2008'), default='mq2007')
  parser.add_argument('--all-objectives',
                      help='To run training with all objective functions',
                      action="store_true", default=False)
  parser.add_argument('--listnet',
                      help='To run training with listnet cross-entropy divergence',
                      action="store_true", default=False)
  parser.add_argument('--kl-divergence',
                      help='To run training with KL divergence',
                      action="store_true", default=False)
  parser.add_argument('--alpha-divergence',
                      help='To run training with alpha divergence',
                      action="store_true", default=False)
  parser.add_argument('--weighted-kl-divergence',
                      help='To run training with weighted KL divergence',
                      action="store_true", default=False)
  return parser.parse_args()

def main(opts):
  """The main function"""

  if opts.all_objectives:
    opts.listnet = True
    opts.kl_divergence = True
    opts.alpha_divergence = True
    opts.weighted_kl_divergence = True

  # Seed for reproducibility
  torch.manual_seed(2019)
  np.random.seed(2019)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  listnet_mq2007_ndcgs = []
  listnet_mq2007_errs = []
  listnet_mq2007_ks = None
  if opts.listnet and opts.data_set == 'mq2007':
    for idx in range(opts.repeat):
      print(f"=====ListNet  MQ2007 ({idx})=======")
      list_net_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
      listnet_mq2007_ks, ndcg, err = list_net_mq2007_training.train(
        criteria.ListNetTopOneCriterion())
      listnet_mq2007_ndcgs.append(ndcg)
      listnet_mq2007_errs.append(err)

  listnet_mq2008_ndcgs = []
  listnet_mq2008_errs = []
  listnet_mq2008_ks = None
  if opts.listnet and opts.data_set == 'mq2008':
    for idx in range(opts.repeat):
      print(f"=====ListNet  MQ2008 ({idx})=======")
      list_net_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
      listnet_mq2008_ks, ndcg, err = list_net_mq2008_training.train(
        criteria.ListNetTopOneCriterion())
      listnet_mq2008_ndcgs.append(ndcg)
      listnet_mq2008_errs.append(err)

  kl_mq2007_ndcgs = []
  kl_mq2007_errs = []
  kl_mq2007_ks = None
  if opts.kl_divergence and opts.data_set == 'mq2007':
    for idx in range(opts.repeat):
      print(f"=====KLDivergence  MQ2007 ({idx})=======")
      kl_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
      kl_mq2007_ks, ndcg, err = kl_divergence_mq2007_training.train(
        criteria.KLDivergenceTopOneCriterion())
      kl_mq2007_ndcgs.append(ndcg)
      kl_mq2007_errs.append(err)

  kl_mq2008_ndcgs = []
  kl_mq2008_errs = []
  kl_mq2008_ks = None
  if opts.kl_divergence and opts.data_set == 'mq2008':
    for idx in range(opts.repeat):
      print(f"=====KLDivergence  MQ2008 ({idx})=======")
      kl_divergence_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
      kl_mq2008_ks, ndcg, err = kl_divergence_mq2008_training.train(
        criteria.KLDivergenceTopOneCriterion())
      kl_mq2008_ndcgs.append(ndcg)
      kl_mq2008_errs.append(err)

  alpha_mq2007_ndcgs = {}
  alpha_mq2007_errs = {}
  alpha_mq2007_vals = [-0.5, 0.5, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 5.0]
  if opts.alpha_divergence and opts.data_set == 'mq2007':
    for alpha in alpha_mq2007_vals:
      alpha_mq2007_ndcgs[alpha] = []
      alpha_mq2007_errs[alpha] = []
      alpha_ks = None
      for idx in range(opts.repeat):
        print(f"=====Alpha Divergence MQ2007 (alpha={alpha})[{idx}]=======")
        alpha_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
        alpha_ks, ndcg, err = alpha_divergence_mq2007_training.train(
          criteria.AlphaDivergenceTopOneCriterion(alpha=alpha))
        alpha_mq2007_ndcgs[alpha].append(ndcg)
        alpha_mq2007_errs[alpha].append(err)

  alpha_mq2008_ndcgs = {}
  alpha_mq2008_errs = {}
  alpha_mq2008_vals = [-0.5, 0.5, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 5.0]
  if opts.alpha_divergence and opts.data_set == 'mq2008':
    for alpha in alpha_mq2008_vals:
      alpha_mq2008_ndcgs[alpha] = []
      alpha_mq2008_errs[alpha] = []
      alpha_ks = None
      for idx in range(opts.repeat):
        print(f"=====Alpha Divergence MQ2008 (alpha={alpha})[{idx}]=======")
        alpha_divergence_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
        alpha_ks, ndcg, err = alpha_divergence_mq2008_training.train(
          criteria.AlphaDivergenceTopOneCriterion(alpha=alpha))
        alpha_mq2008_ndcgs[alpha].append(ndcg)
        alpha_mq2008_errs[alpha].append(err)

  lambda_mq2007_ndcgs = {}
  lambda_mq2007_errs = {}
  lambda_mq2007_vals = [0.1, 0.7, 1.4, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
  if opts.weighted_kl_divergence and opts.data_set == 'mq2007':
    for lambd in lambda_mq2007_vals:
      lambda_mq2007_ndcgs[lambd] = []
      lambda_mq2007_errs[lambd] = []
      lambda_ks = None
      for idx in range(opts.repeat):
        print(f"=====Weighted Divergence  MQ2007 (lambda={lambd})[{idx}]=======")
        lambda_kl_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
        lambda_ks, ndcg, err = lambda_kl_divergence_mq2007_training.train(
          criteria.WeightedKLDivergenceTopOneCriterion(lambd=lambd))
        lambda_mq2007_ndcgs[lambd].append(ndcg)
        lambda_mq2007_errs[lambd].append(err)

  lambda_mq2008_ndcgs = {}
  lambda_mq2008_errs = {}
  lambda_mq2008_vals = [0.1, 0.7, 1.4, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
  if opts.weighted_kl_divergence and opts.data_set == 'mq2008':
    for lambd in lambda_mq2008_vals:
      lambda_mq2008_ndcgs[lambd] = []
      lambda_mq2008_errs[lambd] = []
      lambda_ks = None
      for idx in range(opts.repeat):
        print(f"=====Weighted Divergence  MQ2008 (lambda={lambd})[{idx}]=======")
        lambda_kl_divergence_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
        lambda_ks, ndcg, err = lambda_kl_divergence_mq2008_training.train(
          criteria.WeightedKLDivergenceTopOneCriterion(lambd=lambd))
        lambda_mq2008_ndcgs[lambd].append(ndcg)
        lambda_mq2008_errs[lambd].append(err)

  # print("=====ListMLE  MQ2007=======")
  # list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  # list_mle_mq2007_training.train(criteria.ListMLECriterion())
  # print("=====ListMLE  MQ2008=======")
  # list_mle_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
  # list_mle_mq2008_training.train(criteria.ListMLECriterion())

  # print("=====StatRank  MQ2007=======")
  # list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  # list_mle_mq2007_training.train(criteria.StatRankCriterion())

  ndcgs_list = []
  errs_list = []
  name_list = []
  k_vals = []
  if opts.listnet and opts.data_set == 'mq2007':
    print(f"======ListNet MQ2007 metrics average of [{opts.repeat}] runs======")
    k_vals, listnet_ndcgs, listnet_errs = training.dump_metrics(
      listnet_mq2007_ks, listnet_mq2007_ndcgs, listnet_mq2007_errs, column_head='Try')
    name_list.append('ListN')
    ndcgs_list.append(listnet_ndcgs)
    errs_list.append(listnet_errs)
  if opts.kl_divergence and opts.data_set == 'mq2007':
    print(f"======KLDivergence MQ2007 metrics average of [{opts.repeat}] runs======")
    _, kl_ndcgs, kl_errs = training.dump_metrics(
      kl_mq2007_ks, kl_mq2007_ndcgs, kl_mq2007_errs, column_head='Try')
    name_list.append('KLDiv')
    ndcgs_list.append(kl_ndcgs)
    errs_list.append(kl_errs)
  if opts.alpha_divergence and opts.data_set == 'mq2007':
    for alpha in alpha_mq2007_vals:
      print(f"======Alpha[{alpha}] Divergence MQ2007 [{opts.repeat}] runs======")
      _, alpha_div_ndcgs, alpha_div_errs = training.dump_metrics(
        alpha_ks, alpha_mq2007_ndcgs[alpha], alpha_mq2007_errs[alpha], column_head='Try')
      name_list.append(f"Alph{alpha:.1f}")
      ndcgs_list.append(alpha_div_ndcgs)
      errs_list.append(alpha_div_errs)
  if opts.weighted_kl_divergence and opts.data_set == 'mq2007':
    for lambd in lambda_mq2007_vals:
      print(f"======Lambda[{lambd}] KL Divergence MQ2007 [{opts.repeat}] runs======")
      _, weighted_kl_ndcgs, weighted_kl_errs = training.dump_metrics(
        lambda_ks, lambda_mq2007_ndcgs[lambd], lambda_mq2007_errs[lambd], column_head='Try')
      name_list.append(f"Wgt{lambd:.1f}")
      ndcgs_list.append(weighted_kl_ndcgs)
      errs_list.append(weighted_kl_errs)

  if opts.listnet and opts.data_set == 'mq2008':
    print(f"======ListNet MQ2008 metrics average of [{opts.repeat}] runs======")
    _, listnet_ndcgs, listnet_errs = training.dump_metrics(
      listnet_mq2008_ks, listnet_mq2008_ndcgs, listnet_mq2008_errs, column_head='Try')
    name_list.append('ListNet')
    ndcgs_list.append(listnet_ndcgs)
    errs_list.append(listnet_errs)
  if opts.kl_divergence and opts.data_set == 'mq2008':
    print(f"======KLDivergence MQ2007 metrics average of [{opts.repeat}] runs======")
    _, kl_ndcgs, kl_errs = training.dump_metrics(
      kl_mq2008_ks, kl_mq2008_ndcgs, kl_mq2008_errs, column_head='Try')
    name_list.append('KLDiv')
    ndcgs_list.append(kl_ndcgs)
    errs_list.append(kl_errs)
  if opts.alpha_divergence and opts.data_set == 'mq2008':
    for alpha in alpha_mq2008_vals:
      print(f"======Alpha[{alpha}] Divergence MQ2008 [{opts.repeat}] runs======")
      _, alpha_div_ndcgs, alpha_div_errs = training.dump_metrics(
        alpha_ks, alpha_mq2008_ndcgs[alpha], alpha_mq2008_errs[alpha],
        column_head='Try')
      name_list.append(f"Alph{alpha:.1f}")
      ndcgs_list.append(alpha_div_ndcgs)
      errs_list.append(alpha_div_errs)
  if opts.weighted_kl_divergence and opts.data_set == 'mq2008':
    for lambd in lambda_mq2008_vals:
      print(f"======Lambda[{lambd}] KL Divergence MQ2008 [{opts.repeat}] runs======")
      _, weighted_kl_ndcgs, weighted_kl_errs = training.dump_metrics(
        lambda_ks, lambda_mq2008_ndcgs[lambd], lambda_mq2008_errs[lambd], column_head='Try')
      name_list.append(f"Wgt{lambd:.1f}")
      ndcgs_list.append(weighted_kl_ndcgs)
      errs_list.append(weighted_kl_errs)

  print(f"==========Dataset {opts.data_set.upper()} NDCGs===========")
  print("".join([f"Func,\t"] +
                ['NDCG@' + str(k) + ',' + (' ' if k < 10 else '') for k in k_vals]))
  for idx, ndcgs in enumerate(ndcgs_list):
    print(",\t".join([name_list[idx]] + ["{:.4f}".format(score) for score in ndcgs]))
  print(f"==========Dataset {opts.data_set.upper()} ERRs ===========")
  print(",\t".join([f"Func"] +
                   ['ERR@' + str(k) for k in k_vals]))
  for idx, errs in enumerate(errs_list):
    print(",\t".join([name_list[idx]] + ["{:.4f}".format(score) for score in errs]))

if __name__ == "__main__":
  OPTS = parse_args()
  main(OPTS)
