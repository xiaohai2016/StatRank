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
  parser.add_argument('--log', help='specify the log file location for storing outputs')
  parser.add_argument('--repeat',
                      help='the number of repeating runs to collect the metrics average',
                      type=int, default=10)
  return parser.parse_args()

def main(opts):
  """The main function"""
  if opts.log is not None:
    sys.stdout = open(opts.log, "w")
  else:
    sys.stdout = sys.__stdout__

  # print("=====ListNet  MQ2007=======")
  # listnet_ndcgs = []
  # listnet_errs = []
  # listnet_ks = None
  # for _ in range(opts.repeat):
  #   list_net_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  #   listnet_ks, ndcg, err = list_net_mq2007_training.train(criteria.ListNetTopOneCriterion())
  #   listnet_ndcgs.append(ndcg)
  #   listnet_errs.append(err)
  # print(f"======ListNet MQ2007 metrics average of [{opts.repeat}] runs======")
  # training.dump_metrics(listnet_ks, listnet_ndcgs, listnet_errs, column_head='Try')

  # print("=====ListNet  MQ2008=======")
  # list_net_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
  # list_net_mq2008_training.train(criteria.ListNetTopOneCriterion())

  # print("=====KLDivergence  MQ2007=======")
  # kl_ndcgs = []
  # kl_errs = []
  # kl_ks = None
  # for _ in range(opts.repeat):
  #   kl_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  #   kl_ks, ndcg, err = kl_divergence_mq2007_training.train(criteria.KLDivergenceTopOneCriterion())
  #   kl_ndcgs.append(ndcg)
  #   kl_errs.append(err)
  # print(f"======KLDivergence MQ2007 metrics average of [{opts.repeat}] runs======")
  # training.dump_metrics(kl_ks, kl_ndcgs, kl_errs, column_head='Try')

  print("=====Alpha Divergence  MQ2007=======")
  alpha_ndcgs = {}
  alpha_errs = {}
  alpha_vals = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.5, 2.0, 5.0]
  for alpha in alpha_vals:
    alpha_ndcgs[alpha] = []
    alpha_errs[alpha] = []
    alpha_ks = None
    for _ in range(opts.repeat):
      alpha_divergence_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
      alpha_ks, ndcg, err = alpha_divergence_mq2007_training.train(
        criteria.AlphaDivergenceTopOneCriterion(alpha=alpha))
      alpha_ndcgs[alpha].append(ndcg)
      alpha_errs[alpha].append(err)
  for alpha in alpha_vals:
    print(f"======Alpha[{alpha}] Divergence MQ2007 [{opts.repeat}] runs======")
    training.dump_metrics(alpha_ks, alpha_ndcgs[alpha], alpha_errs[alpha], column_head='Try')

  # print("=====ListMLE  MQ2007=======")
  # list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  # list_mle_mq2007_training.train(criteria.ListMLECriterion())
  # print("=====ListMLE  MQ2008=======")
  # list_mle_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
  # list_mle_mq2008_training.train(criteria.ListMLECriterion())

  # print("=====StatRank  MQ2007=======")
  # list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  # list_mle_mq2007_training.train(criteria.StatRankCriterion())

  sys.stdout = sys.__stdout__

if __name__ == "__main__":
  OPTS = parse_args()
  main(OPTS)
