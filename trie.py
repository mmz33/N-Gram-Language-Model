from queue import Queue

class Trie:
  """Trie data structure used to store ngrams and their frequencies

    Using trie, we can do queries such as finding ngrams, extracting counts, etc
    efficiently. Note also if you add trigrams for example, you are also able to extract information
    about smaller grams such as bigrams since prefix information is also stored
  """

  def __init__(self):
    self.children = {} # dict to store word indexes to represent ngrams
    self.freq = 0

  def add_ngram(self, ngram):
    """Add an ngram to the Trie

    :param ngram: A list of word indexes representing an ngram
    """

    self.freq += 1
    if len(ngram) == 0: return
    idx = ngram[0]
    if idx not in self.children:
      self.children[idx] = Trie()
    next_node = self.children[idx]
    next_node.add_ngram(ngram[1:])

  def get_ngram_freq(self, ngram):
    """Retrieve the frequency of the given ngram

    :param ngram: A list of word indexes representing an ngram
    :return: An int, the frequency of the given ngram
    """

    if len(ngram) == 0: return self.freq
    idx = ngram[0]
    if idx not in self.children:
      return 0 # trigram not found so freq is 0
    next_node = self.children[idx]
    return next_node.get_ngram_freq(ngram[1:])

  def get_children(self):
    """Return the children of the calling node"""

    return self.children

  def get_num_of_children(self):
    """Return the number of children of the calling node"""

    return len(self.children)

if __name__ == '__main__':
  trie = Trie()
  trie.add_ngram([1, 2, 3])
  trie.add_ngram([1, 2, 3])
  trie.add_ngram([1, 2, 5])
  trie.add_ngram([1, 2, 10])
  print('freq: ', trie.get_ngram_freq([1, 2, 3]))
