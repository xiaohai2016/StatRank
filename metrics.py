"""
Ranking performance metrics
"""
import numpy as np


def ranking_precision_score(y_true, y_score, k=10):
  """
  Precision at rank k
  Parameters
  ----------
  y_true : array-like, shape = [n_samples]
      Ground truth (true relevance labels).
  y_score : array-like, shape = [n_samples]
      Predicted scores.
  k : int
      Rank.
  Returns
  -------
  precision @k : float
  """
  unique_y = np.unique(y_true)

  if len(unique_y) > 2:
    raise ValueError("Only supported for two relevance levels.")

  pos_label = unique_y[1]
  n_pos = np.sum(y_true == pos_label)

  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])
  n_relevant = np.sum(y_true == pos_label)

  # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
  return float(n_relevant) / min(n_pos, k)

def average_precision_score(y_true, y_score, k=10):
  """Average precision at rank k
  Parameters
  ----------
  y_true : array-like, shape = [n_samples]
      Ground truth (true relevance labels).
  y_score : array-like, shape = [n_samples]
      Predicted scores.
  k : int
      Rank.
  Returns
  -------
  average precision @k : float
  """
  unique_y = np.unique(y_true)

  if len(unique_y) > 2:
    raise ValueError("Only supported for two relevance levels.")

  pos_label = unique_y[1]
  n_pos = np.sum(y_true == pos_label)

  order = np.argsort(y_score)[::-1][:min(n_pos, k)]
  y_true = np.asarray(y_true)[order]

  score = 0
  for i, _ in enumerate(y_true):
    if y_true[i] == pos_label:
      # Compute precision up to document i
      # i.e, percentage of relevant documents up to document i.
      prec = 0
      for j in range(0, i + 1):
        if y_true[j] == pos_label:
          prec += 1.0
      prec /= (i + 1.0)
      score += prec

  if n_pos == 0:
    return 0

  return score / n_pos

def dcg_score(y_true, y_score, k=10, gains="exponential"):
  """Discounted cumulative gain (DCG) at rank k
  Parameters
  ----------
  y_true : array-like, shape = [n_samples]
      Ground truth (true relevance labels).
  y_score : array-like, shape = [n_samples]
      Predicted scores.
  k : int
      Rank.
  gains : str
      Whether gains should be "exponential" (default) or "linear".
  Returns
  -------
  DCG @k : float
  """
  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])

  if gains == "exponential":
    gains = 2 ** y_true - 1
  elif gains == "linear":
    gains = y_true
  else:
    raise ValueError("Invalid gains option.")

  # highest rank is 1 so +2 instead of +1
  discounts = np.log2(np.arange(len(y_true)) + 2)
  return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
  """Normalized discounted cumulative gain (NDCG) at rank k
  Parameters
  ----------
  y_true : array-like, shape = [n_samples]
      Ground truth (true relevance labels).
  y_score : array-like, shape = [n_samples]
      Predicted scores.
  k : int
      Rank.
  gains : str
      Whether gains should be "exponential" (default) or "linear".
  Returns
  -------
  NDCG @k : float
  """
  best = dcg_score(y_true, y_true, k, gains)
  actual = dcg_score(y_true, y_score, k, gains)
  return 0 if not best else actual / best


def err(y_true, y_score, k=10, max_grade=2):
  """
  ERR (Expected Reciprocal Rank)
  Parameters
  ----------
  y_true : array-like, shape = [n_samples]
      Ground truth (true relevance labels).
  y_score : array-like, shape = [n_samples]
      Predicted scores.
  k : int
      Rank.
  max_grade: int, maximum relevance score,
             max_grade should be *2*
  """
  if k is None:
    k = len(y_true)

  order = np.argsort(y_score)[::-1]
  y_true = np.take(y_true, order[:k])
  ranking = map(float, y_true)

  result = 0.0
  prob_step_down = 1.0

  for rank, rel in enumerate(ranking):
    rank += 1
    utility = (pow(2, rel) - 1) / pow(2, max_grade)
    result += prob_step_down * utility / rank
    prob_step_down *= (1 - utility)

  return result
