import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

def plot_kv_iterable(iter_data, xlabel, ylabel, log10_space=True):
  """Plot the given iterable data structure where keys are represented by
  the x-axis in ascending order and values are represented by the y-axis

  :param iter_data: An iterable data structure, a dict or list
  :param xlabel: A string, x-axis label name
  :param ylabel: A string, y-axis label name
  :param log10_space: A boolean, if True use log10 for y-axis and nothing otherwise
  """

  assert isinstance(iter_data, dict) or isinstance(iter_data, list), 'iter_data should be a dict or list'

  iter_data = sorted(iter_data.items() if isinstance(iter_data, dict) else iter_data, key=lambda kv: kv[0])
  x = []
  y = []
  for k, v in iter_data:
    x.append(int(k))
    y.append(np.log10(float(v)) if log10_space else v)
  plt.xticks(np.arange(0, max(x), 5), rotation=90)
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def plot_count_of_counts(d, n):
  """Plot count of counts distribution

  :param d: A list, the elements are pairs of ngrams and their freq
  """

  res = defaultdict(int)
  for _, cnt in d:
    res[cnt] += 1

  plot_kv_iterable(res,
                   xlabel=str(n) + '-gram count',
                   ylabel='count of counts')

  # res = defaultdict(int)
  # with open(file_path, 'r') as read_file:
  #   for line in read_file:
  #     x = line.strip().split()
  #     res[x[-1]] += 1
  # plot_kv_iterable(res,
  #                  xlabel=str(n) + '-gram count',
  #                  ylabel='count of counts')

def get_top_k_freq_items(d, k):
  """Return the top k elements w.r.t to their values (freq)

  :param d: A dict or list, values are frequencies
  :param k: An integer, the number of top elements
  :return: A list containing the top k freq elements
  """

  if isinstance(d, dict):
    return Counter(d).most_common(k)
  elif isinstance(d, list):
    return sorted(d, key=lambda kv: kv[1], reverse=True)[:k]
  else:
    raise TypeError('d should be either a dict or list')

def write_kv_iter_to_file(iter_data, out_path=None):
  """Write kv pair dict or list to file

  :param iter_data: dict or list
  :param out_path: A string, the output path
  """

  assert isinstance(iter_data, dict) or isinstance(iter_data, list), 'iter_data should be a list or dict'
  assert out_path is not None, 'please specify the output path'
  with open(out_path, 'w') as out_file:
    for k, v in iter_data.items() if isinstance(iter_data, dict) else iter_data:
      out_file.write(str(k) + ' ' + str(v) + '\n')