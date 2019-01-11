from collections import defaultdict

class IndexMap:
  """Data structure for indexing words by unique ids
  It allows retrieve queries in both direction (wrd->idx, and idx->wrd)
  """

  def __init__(self, vocabs_file=None):
    """
    :param vocabs_file: A string, vocabs file path to load
    """

    self.idx = 0 # unique index for each word

    self.wrd_to_idx = {}
    self.idx_to_wrd = {}
    self.wrd_freq = defaultdict(int)

    if vocabs_file:
      with open(vocabs_file, 'r') as vocabs:
        for wrd in vocabs:
          self.add_wrd(wrd.strip())
    else:
      # these symbols already exist in the vocabulary
      for wrd in {'<s>', '</s>', '<unk>'}:
        self.add_wrd(wrd)

  @staticmethod
  def get_unk_wrd():
    return '<unk>'

  @staticmethod
  def get_start_wrd():
    return '<s>'

  @staticmethod
  def get_end_wrd():
    return '</s>'

  def get_unk_id(self):
    return self.wrd_to_idx[self.get_unk_wrd()]

  def get_start_id(self):
    return self.wrd_to_idx[self.get_start_wrd()]

  def get_end_id(self):
    return self.wrd_to_idx[self.get_end_wrd()]

  def add_wrd(self, wrd):
    """Update index maps and increase word's frequency

    :param wrd: A string, the input word
    """

    if wrd not in self.wrd_to_idx:
      self.wrd_to_idx[wrd] = self.idx
      self.idx_to_wrd[self.idx] = wrd
      self.idx += 1

    self.wrd_freq[self.wrd_to_idx[wrd]] += 1

  def get_wrd_by_idx(self, idx):
    """Return the word of the given index

    :param idx: An int, the index of the given word
    :return: A string, <unk> symbol if index does not exist else the word of the given index
    """

    if idx not in self.idx_to_wrd:
      return self.get_unk_wrd()
    return self.idx_to_wrd[idx]

  def get_idx_by_wrd(self, wrd):
    """Return the index of the given word

    :param wrd: A string, the word of the given index
    :return: An integer, -1 if word does not exist else the index of the word
    """

    if wrd not in self.wrd_to_idx:
      return -1
    return self.wrd_to_idx[wrd]

  def get_unique_wrd_count(self):
    """Return the number of unique words

    :return: An integer, the number of unique words
    """

    return self.idx

  def get_wrd_freq(self, idx):
    """Return the frequency of a word given it's index

    :param idx: An integer, an index of a word
    :return: An integer, the frequency of the given word
    """

    return self.wrd_freq[idx]

  def get_wrd_freq_items(self):
    """Return word-freq pair list"""

    return self.wrd_freq.items()