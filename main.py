"""
The main entry point - handles command line options and invocations
"""
import argparse
import sys
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
  listnet_mq2007_ndcgs = []
  listnet_mq2007_errs = []
  listnet_mq2007_ks = None
  if opts.listnet and opts.data_set == 'mq2007':
    for idx in range(opts.repeat):
      print(f"=====ListNet  MQ2007 ({idx})=======")
      list_net_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
      listnet_mq2007_ks, ndcg, err = list_net_mq2007_training.train(criteria.ListNetTopOneCriterion())
      listnet_mq2007_ndcgs.append(ndcg)
      listnet_mq2007_errs.append(err)

  listnet_mq2008_ndcgs = []
  listnet_mq2008_errs = []
  listnet_mq2008_ks = None
  if opts.listnet and opts.data_set == 'mq2008':
    for idx in range(opts.repeat):
      print(f"=====ListNet  MQ2008 ({idx})=======")
      list_net_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
      listnet_mq2008_ks, ndcg, err = list_net_mq2008_training.train(criteria.ListNetTopOneCriterion())
      listnet_mq2008_ndcgs.append(ndcg)
      listnet_mq2008_errs.append(err)

  kl_mq2007_ndcgs = []
  kl_mq2007_errs = []
  kl_mq2007_ks = None
  if opts.kl_divergence and opts.data_set == 'mq2007':
    for idx in range(opts.repeat):
      print(f"=====KLDivergence  MQ2007 ({idx})=======")
      kl_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
      kl_mq2007_ks, ndcg, err = kl_divergence_mq2007_training.train(criteria.KLDivergenceTopOneCriterion())
      kl_mq2007_ndcgs.append(ndcg)
      kl_mq2007_errs.append(err)

  kl_mq2008_ndcgs = []
  kl_mq2008_errs = []
  kl_mq2008_ks = None
  if opts.kl_divergence and opts.data_set == 'mq2008':
    for idx in range(opts.repeat):
      print(f"=====KLDivergence  MQ2008 ({idx})=======")
      kl_divergence_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
      kl_mq2008_ks, ndcg, err = kl_divergence_mq2008_training.train(criteria.KLDivergenceTopOneCriterion())
      kl_mq2008_ndcgs.append(ndcg)
      kl_mq2008_errs.append(err)

  alpha_mq2007_ndcgs = {}
  alpha_mq2007_errs = {}
  alpha_mq2007_vals = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.5, 2.0, 5.0]
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
  alpha_mq2008_vals = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.5, 2.0, 5.0]
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
  lambda_mq2007_vals = [0.1, 0.7, 0.9, 1.1, 1.3, 2.0, 5.0]
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
  lambda_mq2008_vals = [0.1, 0.7, 0.9, 1.1, 1.3, 2.0, 5.0]
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

  if opts.listnet and opts.data_set == 'mq2007':
    print(f"======ListNet MQ2007 metrics average of [{opts.repeat}] runs======")
    training.dump_metrics(listnet_mq2007_ks, listnet_mq2007_ndcgs, listnet_mq2007_errs, column_head='Try')
  if opts.kl_divergence and opts.data_set == 'mq2007':
    print(f"======KLDivergence MQ2007 metrics average of [{opts.repeat}] runs======")
    training.dump_metrics(kl_mq2007_ks, kl_mq2007_ndcgs, kl_mq2007_errs, column_head='Try')
  if opts.alpha_divergence and opts.data_set == 'mq2007':
    for alpha in alpha_mq2007_vals:
      print(f"======Alpha[{alpha}] Divergence MQ2007 [{opts.repeat}] runs======")
      training.dump_metrics(alpha_ks, alpha_mq2007_ndcgs[alpha], alpha_mq2007_errs[alpha], column_head='Try')
  if opts.weighted_kl_divergence and opts.data_set == 'mq2007':
    for lambd in lambda_mq2007_vals:
      print(f"======Lambda[{lambd}] KL Divergence MQ2007 [{opts.repeat}] runs======")
      training.dump_metrics(lambda_ks, lambda_mq2007_ndcgs[lambd], lambda_mq2007_errs[lambd], column_head='Try')

  if opts.listnet and opts.data_set == 'mq2008':
    print(f"======ListNet MQ2008 metrics average of [{opts.repeat}] runs======")
    training.dump_metrics(listnet_mq2008_ks, listnet_mq2008_ndcgs, listnet_mq2008_errs, column_head='Try')
  if opts.kl_divergence and opts.data_set == 'mq2008':
    print(f"======KLDivergence MQ2007 metrics average of [{opts.repeat}] runs======")
    training.dump_metrics(kl_mq2008_ks, kl_mq2008_ndcgs, kl_mq2008_errs, column_head='Try')
  if opts.alpha_divergence and opts.data_set == 'mq2008':
    for alpha in alpha_mq2008_vals:
      print(f"======Alpha[{alpha}] Divergence MQ2008 [{opts.repeat}] runs======")
      training.dump_metrics(alpha_ks, alpha_mq2008_ndcgs[alpha], alpha_mq2008_errs[alpha], column_head='Try')
  if opts.weighted_kl_divergence and opts.data_set == 'mq2008':
    for lambd in lambda_mq2008_vals:
      print(f"======Lambda[{lambd}] KL Divergence MQ2008 [{opts.repeat}] runs======")
      training.dump_metrics(lambda_ks, lambda_mq2008_ndcgs[lambd], lambda_mq2008_errs[lambd], column_head='Try')

if __name__ == "__main__":
  OPTS = parse_args()
  main(OPTS)
