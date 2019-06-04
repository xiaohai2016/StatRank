"""Data anlysis and visualization script(s)"""
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import click

from data_loader import load_microsoft_dataset

def tsne_fit(feature_vectors, n_components=2, verbose=1, perplexity=10, n_iter=300):
  """Use t-SNE algorithm to perform dimension reduction for feature vectors"""
  tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
  return tsne.fit_transform(feature_vectors)

def pca_fit(feature_vectors, n_components=2):
  """Use PCA algorithm to perform dimension reduction for feature vectors"""
  pca = PCA(n_components=n_components)
  return pca.fit_transform(feature_vectors)

@click.command()
@click.option('--dataset', default='MQ2007', help='MQ2007 or MQ2008')
@click.option('--fold', default=1, help='1 to 5')
@click.option('--algo', default='pca', help='tsne or pca')
@click.option('--plot-count', default=10)
@click.option('--batch-count', default=10)
def plot_analysis(dataset, fold, algo, plot_count, batch_count):
  """Use t-SNE/PCA algorithm to plot feature vectors"""
  path = f"resources/{dataset}/Fold{fold}/train.txt"
  data = load_microsoft_dataset(path, feature_count=46)
  data_len = len(data)
  if plot_count < 0:
    plot_count = (data_len // batch_count) + 1
  for tries in range(plot_count):
    all_rel_scores = None
    all_feature_vectors = None
    for idx in range(batch_count):
      data_idx = idx + tries * batch_count
      if data_idx >= data_len:
        break
      _, rel_scores, feature_vectors = data[data_idx]
      if all_rel_scores is None:
        all_rel_scores = rel_scores.astype(int)
        all_feature_vectors = feature_vectors
      else:
        all_rel_scores = np.concatenate((all_rel_scores, rel_scores.astype(int)), axis=0)
        all_feature_vectors = np.concatenate((all_feature_vectors, feature_vectors), axis=0)
    if all_rel_scores is None:
      break

    results = tsne_fit(all_feature_vectors) if algo == "tsne" else pca_fit(all_feature_vectors)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
      x=results[:, 0], y=results[:, 1],
      hue=all_rel_scores,
      size=all_rel_scores,
      legend="full"
    )
    plt.show()

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
  plot_analysis()
