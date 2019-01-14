from collections import defaultdict
from index_map import IndexMap
import utils
from trie import Trie
from queue import Queue
import numpy as np

# data paths
data_train = 'data/train.corpus'
data_test = 'data/test.corpus'
vocabulary_path = 'data/vocabulary'

class LM:
  """Represents the main entry point of the language model"""

  def __init__(self, vocabs_file=None):
    self.corpus = IndexMap()
    if vocabs_file:
      self.vocabs = IndexMap(vocabs_file)

    self.running_wrds_num = 0
    self.sent_num = 0
    self.sent_len_sum = 0
    self.sent_len_to_freq = defaultdict(int)
    self.prepare_lm()
    self.avg_sent_len = (1.0 * self.sent_len_sum)/self.sent_num

    self.unk_cnt = 0
    self.oov = 0.0 # out-of-vocabulary rate

    self.ngrams_root = Trie() # ngrams trie root

    self.b = [] # contain b discounting params of (index+1)-gram
    self.compute_b(2, vocabulary=self.corpus)
    print('Discounting parameters:')
    for n in range(len(self.b)):
      print('b_{} = {}'.format(n+1, self.b[n]))

  def prepare_lm(self):
    """Prepare the language model for analysis and computations"""

    print('Reading corpus for analysis...')

    with open(data_train, 'r') as read_corpus:
      for line in read_corpus:
        words = line.strip().split(' ')
        sent_len = len(words)
        self.sent_len_sum += sent_len
        self.sent_len_to_freq[sent_len] += 1
        self.sent_num += 1
        for wrd in words:
          self.corpus.add_wrd(wrd)
          self.running_wrds_num += 1

  ################################ Ex1 (a)-(b)-(c) ################################

  def show_corpus_analysis(self):
    """Show corpus data analysis"""

    print('Number of running words: %d' % self.running_wrds_num)
    print('Number of sentences: %d' % self.sent_num)
    print('Average sentence length: %.2f' % self.avg_sent_len)

    utils.plot_kv_iterable(self.sent_len_to_freq,
                           xlabel='Sentence length',
                           ylabel='Frequency',
                           xticks=5)

  ################################ Ex2 (a) ################################

  def get_word_freq_dict(self):
    """Return a dict where key is a word and value is it's frequency

      It transforms the wrd_freq dict from index-based key to word-based key
      so that it can be used for retrieving the top k frequent words

    :return: A dict, word to freq mapping
    """

    res = {}
    for idx, freq in self.corpus.get_wrd_freq_items():
      wrd = self.corpus.get_wrd_by_idx(idx)
      res[wrd] = freq
    return res

  @staticmethod
  def get_top_10_freq_words():
    """Return the top 10 frequent words

    :return: A list, top 10 frequent words
    """

    return utils.get_top_k_freq_items(lm.get_word_freq_dict(), k=10)

  ############################# Ex2 (b)-(c)-(d)-(e) ####################################

  def get_corpus_tokens(self, vocabulary):
    """Read corpus and store the tokens in a list

    If with_unk is True, then unk_cnt is used to store the number
    of OOV words to compute it's rate later

    :param vocabulary: IndexMap, either corpus or vocabs
    :return: A list of word tokens with start and end symbols
    """

    tokens = []
    start_id = vocabulary.get_start_id()
    end_id = vocabulary.get_end_id()
    unk_id = vocabulary.get_unk_id()
    with open(data_train, 'r') as read_corpus:
      for line in read_corpus:
        sent = line.strip().split(' ')
        tokens.append(start_id)
        for wrd in sent:
          idx = vocabulary.get_idx_by_wrd(wrd)
          if idx == unk_id:
            self.unk_cnt += 1
          tokens.append(idx)
        tokens.append(end_id)
    return tokens

  def generate_ngrams(self, n, vocabulary):
    """Generate the ngrams of the corpus and store them in a Trie data structure

    :param n: An integer, the rank of the gram
    :param vocabulary: An IndexMap, either corpus or vocabs
    :return: A Trie representing the ngrams
    """

    print('Reading word tokens from %s' % 'corpus' if vocabulary == self.corpus else 'vocabulary')
    tokens = self.get_corpus_tokens(vocabulary)
    if vocabulary != self.corpus:
      self.oov = (self.unk_cnt / self.running_wrds_num) * 100.0
      print('OOV rate: %.02f %%' % self.oov)

    print('Generating %d-grams from %s' % (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    for i in range(len(tokens)-n+1):
      ngram = tokens[i:i+n]
      self.ngrams_root.add_ngram(ngram)

    print('%d-grams are stored in a Trie' % n)

  def extract_ngrams_and_freq(self, n, vocabulary):
    """Extract ngrams and their frequencies, display top 10 frequent ngrams,
      plot the count of counts distribution

    :param n: An integer, the rank of the gram
    :param vocabulary: An indexMap, either corpus or vocabs
    """

    if self.ngrams_root.is_empty():
      self.generate_ngrams(n, vocabulary)

    print('Extracting %d-grams with their frequencies using %s' % \
          (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    res = self.ngrams_root.bfs(n, vocabulary)

    print('Extraction is done.')

    top_10_ngrams = utils.get_top_k_freq_items(res, k=10)
    print('Top 10 %d-grams:' % n, top_10_ngrams)

    print('Preparing to plot count of counts distribution')
    utils.plot_count_of_counts(res, n)

  ########################### Ex4 ##################################

  def compute_b(self, n, vocabulary):
    """Computes the discounting parameters for up to n-gram
       e.g if n = 2, then it will compute b_uni, and b_bi

    :param n: An integer, the rank of gram
    :param vocabulary: An IndexMap, either corpus or vocabs
    """

    assert self.ngrams_root.get_depth() <= n

    # ngrams are not added to the trie yet
    if self.ngrams_root.is_empty():
      self.generate_ngrams(n, vocabulary)

    q = Queue()
    q.put(self.ngrams_root) # add root
    for i in range(n):
      singeltons = 0
      doubletons = 0
      next_q = Queue()
      while not q.empty():
        u = q.get()
        for idx, child in u.get_children().items():
          next_q.put(child)
          if child.get_freq() == 1:
            singeltons += 1
          elif child.get_freq() == 2:
            doubletons += 1
      self.b.append(singeltons/(singeltons + 2.0 * doubletons))
      q = next_q

  def compute_prob(self, w, h):
    """Computes the bigram probability p(w|h) using absolute discounting with
       interpolation where h is word history

    :param w: An integer, the index of word w
    :param h: A list containing the indexes of word history
    :param vocabulary: An IndexMap, either corpus or vocabs
    :return: A float, p(w|h)
    """

    # backoff to unigram (base case)
    if len(h) == 0:
      prob = self.ngrams_root.get_num_of_children() # W - N_0(.)
      prob /= float(self.vocabs.get_num_of_words() * self.ngrams_root.get_freq()) # W * N
      b_uni = self.b[0]
      prob *= b_uni
      w_node = self.ngrams_root.get_ngram_last_node([w])
      if w_node is not None:
        w_freq = w_node.get_freq()
        prob += max(float(w_freq - b_uni)/self.ngrams_root.get_freq(), 0.0)
      return prob

    h_node = self.ngrams_root.get_ngram_last_node(h)

    # history is not found so backoff
    if h_node is None:
      return self.compute_prob(w, h[1:])

    prob = self.b[len(h)]
    prob *= float(h_node.get_num_of_children()) / h_node.get_freq() # (W - N_0(v,.))/N(v)
    prob *= self.compute_prob(w, h[1:]) # recursively backoff

    # add the first term of the interpolation
    w_node = h_node.get_ngram_last_node([w])
    if w_node is not None:
      w_h_freq = w_node.get_freq()
      prob += max(float(w_h_freq - self.b[len(h)]) / h_node.get_freq(), 0.0)

    return prob

  def verify_normalization(self):
    """Verify the normalization of bigram and unigram probabilities

    :return: True if the probabilities are normalized and False otherwise
    """

    print('Verifying probability normalization...')

    bigram_probs = 0.0
    unigram_probs = 0.0
    for w in range(0, self.vocabs.get_num_of_words()):
      bigram_probs += self.compute_prob(w, [10]) # any word for history
      unigram_probs += self.compute_prob(w, [])

    print('bigram_probs: {}, unigram_probs: {}'.format(bigram_probs, unigram_probs))
    return abs(1.0 - bigram_probs) <= 1e-02 and abs(1.0 - unigram_probs) <= 1e-02

  ########################### Ex5 ##################################

  def perplexity(self):
    """Compute the perplexity of the language model on test corpus

    :return: A float, perplexity of the LM
    """

    print('Computing model perplexity...')

    LL = 0.0
    norm = 0
    with open(data_test, 'r') as test_corpus:
      for line in test_corpus:
        sent = line.strip().split()
        h = self.vocabs.get_start_id()
        for wrd in sent:
          w = self.vocabs.get_idx_by_wrd(wrd)
          prob = self.compute_prob(w, [h])
          LL += np.log(prob)
          h = w
        LL += np.log(self.compute_prob(self.vocabs.get_end_id(), [h]))
        norm += len(sent)+1
    return np.exp(-LL/norm)

#######################################################################

if __name__ == '__main__':
  lm = LM(vocabs_file=vocabulary_path)
  # lm.show_corpus_analysis()
  # lm.extract_ngrams_and_freq(n=3, vocabulary=lm.corpus)
  # lm.extract_ngrams_and_freq(n=3, vocabulary=lm.vocabs)
  if lm.verify_normalization():
    print('Probabilities are normalized!')
  else:
    print('Probabilities are not normalized!')
  print('Test PP: {}'.format(lm.perplexity()))