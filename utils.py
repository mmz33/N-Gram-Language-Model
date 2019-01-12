import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_kv_iterable(iter, xlabel, ylabel, log10_space=False):
  """Plot the given iterable data structure

  :param iter: An iterable data structure, a dict or list
  :param xlabel: A string, x-axis label name
  :param ylabel: A string, y-axis label name
  :param log10_space: A boolean, if True use log10 for y-axis and nothing otherwise
  """

  assert isinstance(iter, dict) or isinstance(iter, list), 'iter should be a dict or list'
  x = []
  y = []
  for k, v in iter.items() if isinstance(iter, dict) else iter:
    x.append(int(k))
    y.append(np.log10(float(v)) if log10_space else v)
  plt.xticks(np.arange(0, max(x), 5), rotation=90)
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def get_top_k_freq_words(d, k):
  """Return the top k elements w.r.t to their values (freq)

  :param d: A dict, values are frequencies
  :param k: An integer, the number of top elements
  :return: A list containing the top k freq elements
  """

  return Counter(d).most_common(k)

def write_kv_iter_to_file(iter, out_path=None):
  """Write kv pair dict or list to file

  :param iter: dict or list
  :param out_path: A string, the output path
  """

  assert isinstance(iter, dict) or isinstance(iter, list), 'iter should be a list or dict'
  assert out_path is not None, 'please specify the output path'
  with open(out_path, 'w') as out_file:
    for k, v in iter.items() if isinstance(iter, dict) else iter:
      out_file.write(str(k) + ' ' + str(v) + '\n')