"""
Utilities for loading datasets
"""
import sys
import csv
import collections
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader


def load_microsoft_dataset(csvfile, max_entries=sys.maxsize,
                           feature_count=46, light_filtering=True):
  """
  To load MQ2007/2008/MSLR-WEB10K/30K CSV file as a dictionary.

  Args:
    csvfile (str): the data file path in csv format.
    max_entries (int): the maximum number of lines to read. Used in testing
                       to avoid loading the entire file.
    feature_count (int): the number of features in each line. MQ2007/MQ2008 has
                         46 features while MSLR-WEB10K/30K has 136 features.
    light_filtering (boolean): if perform light filtering on the data set, i.e.
                               remove queries with less than 10 response documents
                               and with no relevant documents in reponses.

  Returns:
    data_sorted: a list of tuples of (query_id, relevance_scores, feature_vectors),
                 sorted in descending order by relevance_scores.
                 Both relevance_scores and feature_vectors are of numpy.ndarray
                 of shape (num) and (num, feature_count) respectively, where num
                 is the number of response items for the query.
  """
  with open(csvfile, newline='', encoding='utf-8') as hf_csv:
    reader = csv.reader(hf_csv, delimiter=' ')
    data = collections.defaultdict(lambda: [])
    for _, row in zip(range(max_entries), reader):
      relevance = int(row[0])
      qid = int(row[1].split(':')[1])
      features = [itm[1] for itm in sorted(
        [(int(fid), float(val)) for fid, val in \
          [row[i+2].split(':') for i in range(feature_count)]],
        key=lambda tup: tup[0])]
      data[qid].append((relevance, features))
    data_transformed = [(k,) + tuple(
      np.asarray(list(itm), dtype=np.float64) for itm in zip(*v)) for k, v in data.items()]
    if light_filtering:
      return list(filter(lambda tup: len(tup[1]) >= 10 and sum(tup[1]) > 0, data_transformed))
    return data_transformed

def get_ms_dataset(csvfile, feature_count=46, max_entries=sys.maxsize,
                   shuffle=True, scaler_id=None, batch_size=1):
  """Load Microsoft dataset and convert it into a batch data iterator."""
  data = load_microsoft_dataset(csvfile, feature_count=feature_count, max_entries=max_entries)
  if scaler_id is not None:
    scaler = MinMaxScaler() if scaler_id == 'MinMax' else RobustScaler()
    data = [(qid, relevance, scaler.fit_transform(feature)) for qid, relevance, feature in data]
  return DataLoader(data, shuffle=shuffle, batch_size=batch_size)

def get_ms_dataset_preload(data, shuffle=True, scaler_id=None, batch_size=1):
  """Wrap preloaded Microsoft dataset into a batch data iterator."""
  if scaler_id is not None:
    scaler = MinMaxScaler() if scaler_id == 'MinMax' else RobustScaler()
    data = [(qid, relevance, scaler.fit_transform(feature)) for qid, relevance, feature in data]
  return DataLoader(data, shuffle=shuffle, batch_size=batch_size)
