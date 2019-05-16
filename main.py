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
  return parser.parse_args()

def main(opts):
  """The main function"""
  if opts.log is not None:
    sys.stdout = open(opts.log, "w")
  else:
    sys.stdout = sys.__stdout__

  print("=====ListNet  MQ2007=======")
  list_net_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  list_net_mq2007_training.train(criteria.ListNetTopOneCriterion())
  print("=====ListNet  MQ2008=======")
  list_net_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
  list_net_mq2008_training.train(criteria.ListNetTopOneCriterion())

  print("=====ListMLE  MQ2007=======")
  list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  list_mle_mq2007_training.train(criteria.ListMLECriterion())
  print("=====ListMLE  MQ2008=======")
  list_mle_mq2008_training = training.MQ200XTrainer(use_mq2007=False)
  list_mle_mq2008_training.train(criteria.ListMLECriterion())

  # print("=====StatRank  MQ2007=======")
  # list_mle_mq2007_training = training.MQ200XTrainer(use_mq2007=True)
  # list_mle_mq2007_training.train(criteria.StatRankCriterion())

  sys.stdout = sys.__stdout__

if __name__ == "__main__":
  OPTS = parse_args()
  main(OPTS)
