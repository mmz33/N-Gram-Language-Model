import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

def plot_kv_iterable(iter_data, xlabel, ylabel, xticks=None, log10_x=False, log10_y=True):
  """Plot the given iterable data structure where keys are represented by
  the x-axis in ascending order and values are represented by the y-axis

  :param iter_data: An iterable data structure, a dict or list
  :param xlabel: A string, x-axis label name
  :param ylabel: A string, y-axis label name
  :param xticks: An integer, the number of x-axis ticks
  :param log10_x: A boolean, if True use log10 for y-axis
  :param log10_y: A boolean, if True use log10 for x-axis
  """

  assert isinstance(iter_data, dict) or isinstance(iter_data, list), 'iter_data should be a dict or list'

  iter_data = sorted(iter_data.items() if isinstance(iter_data, dict) else iter_data, key=lambda kv: kv[0])
  x = []
  y = []
  for k, v in iter_data:
    x.append(np.log10(float(k)) if log10_x else int(k))
    y.append(np.log10(float(v)) if log10_y else int(v))
  if xticks is not None:
    plt.xticks(np.arange(0, max(x), xticks), rotation=90)
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def plot_count_of_counts(d, n):
  """Plot count of counts distribution

  :param d: A list, the elements are pairs of ngrams and their freq
  :param n: An integer, the rank of the gram
  """

  res = defaultdict(int)
  for _, cnt in d:
    res[cnt] += 1

  plot_kv_iterable(res,
                   xlabel=str(n) + '-gram count',
                   ylabel='count of counts',
                   log10_x=True)

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

  iter_data = sorted(iter_data.items() if isinstance(iter_data, dict) else iter_data, key=lambda kv: kv[0])

  with open(out_path, 'w') as out_file:
    for k, v in iter_data.items() if isinstance(iter_data, dict) else iter_data:
      for kk in k:
        out_file.write(str(kk) + ' ')
      out_file.write(str(v) + '\n')