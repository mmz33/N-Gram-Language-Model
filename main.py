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

    # NOTE: The purpose of creating this dict of roots is only
    # for the following experiment: Comparing the computation of bigrams
    # and unigrams probs using trigram counts vs the computation of bigrams
    # and unigrams probs using their own tries. You should not create multiple
    # trie trees usually since you have prefix information

    self.ngrams_root = {} # contains the root of (key)-gram trie

    self.b = [] # contain b discounting params of (index+1)-gram
    self.compute_b(n=2, vocabulary=self.corpus)
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

    print('Size of vocabulary: %d' % self.vocabs.get_num_of_words())
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
  def get_top_10_freq_words(res):
    """Return the top 10 frequent words

    :return: A list, top 10 frequent words
    """

    return utils.get_top_k_freq_items(res, k=10)

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
    """Generate ngrams from the given vocabulary (corpus or vocabs)
       and store them in a Trie data structure

    :param n: An integer, the rank of the grams that are generated
    :param vocabulary: An IndexMap, either corpus or vocabs
    :return: A Trie representing the ngrams
    """

    start_id = vocabulary.get_start_id()
    end_id = vocabulary.get_end_id()
    unk_id = vocabulary.get_unk_id()

    print('Generating %d-grams from %s' % (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    self.ngrams_root[n] = Trie()
    with open(data_train, 'r') as read_corpus:
      for line in read_corpus:
        sent = line.strip().split(' ')
        ngram = [start_id]
        for wrd in sent:
          if len(ngram) == n:
            self.ngrams_root[n].add_ngram(ngram)
            ngram = ngram[1:]
          idx = vocabulary.get_idx_by_wrd(wrd)
          if idx == unk_id:
            self.unk_cnt += 1
          ngram.append(idx)
        if len(ngram) == n:
          self.ngrams_root[n].add_ngram(ngram)
          ngram = ngram[1:]
        ngram.append(end_id)
        self.ngrams_root[n].add_ngram(ngram)

    print('%d-grams are now stored in a Trie' % n)

    if vocabulary != self.corpus:
      self.oov = (self.unk_cnt / self.running_wrds_num) * 100.0
      print('OOV rate: %.02f %%' % self.oov)

  def extract_ngrams_and_freq(self, n, vocabulary):
    """Extract ngrams and their frequencies, display top 10 frequent ngrams,
      plot the count of counts distribution

    :param n: An integer, the rank of the gram
    :param vocabulary: An indexMap, either corpus or vocabs
    """

    if n not in self.ngrams_root:
      self.generate_ngrams(n, vocabulary)

    print('Extracting %d-grams with their frequencies using %s' % \
          (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    res = self.ngrams_root[n].bfs(n)

    print('Extraction is done.')

    return res

  def get_top_10_ngram_freq(self, n, vocabulary):
    """Return the top 10 frequent ngrams

    :param res: A dict, keys are ngrams and values are counts
    :return: A list of top 10 frequent ngrams
    """

    top_10_ngrams = utils.get_top_k_freq_items(self.extract_ngrams_and_freq(n, vocabulary), k=10)

    # map to words
    res = []
    for kv in top_10_ngrams:
      ngram_wrds = []
      for idx in kv[0]:
        ngram_wrds.append(vocabulary.get_wrd_by_idx(idx))
      res.append((ngram_wrds, kv[1]))
    return res

  ############################# Ex3 ####################################

  def get_summed_counts(self, n, vocabulary):
    """Calculate bi-/uni- grams from trigrams and compare to directly extracted bi-/uni- grams

    :param n: An integer, the rank of the source n-gram
    :param vocabulary: An indexMap, either corpus or vocabs
    """

    assert n >= 3, 'n should be at least 3'

    res_trigram = self.extract_ngrams_and_freq(n, vocabulary)
    res_bi = self.extract_ngrams_and_freq(n-1, vocabulary)
    res_uni = self.extract_ngrams_and_freq(n-2, vocabulary)

    summed_bigrams = defaultdict(int)
    summed_unigrams = defaultdict(int)

    for k, v in res_trigram.items():
      summed_bigrams[k[1:]] += v
      summed_unigrams[k[-1]] += v

    print("bigram difference: ")
    for k, v in res_bi.items():
      if v != summed_bigrams[k]:
        print("bigram: {} extracted value: {} summed value: {}".format(k, v ,summed_bigrams[k]))

    print("unigram difference: ")
    for k, v in res_uni.items():
      if v != summed_unigrams[k]:
        print("unigram: {} extracted value: {} summed value: {}".format(k, v, summed_unigrams[k]))

  ########################### Ex4 ##################################

  def compute_b(self, n, vocabulary):
    """Computes the discounting parameters for up to n-gram
       e.g if n = 2, then it will compute b_uni, and b_bi

    :param n: An integer, the rank of gram
    :param vocabulary: An IndexMap, either corpus or vocabs
    """

    print('Computing discounting parameters...')

    # ngrams are not added to the trie yet
    if n not in self.ngrams_root:
      self.generate_ngrams(n, vocabulary)

    q = Queue()
    q.put(self.ngrams_root[n]) # add root
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

  def compute_prob(self, w, h, n):
    """Computes the bigram probability p(w|h) using absolute discounting with
       interpolation where h is word history

    :param w: An integer, the index of word w
    :param h: A list containing the indexes of word history
    :param n: An integer, the rank of the grams
    :return: A float, p(w|h)
    """

    # backoff to unigram (base case)
    if len(h) == 0:
      prob = self.b[0]
      prob *= self.ngrams_root[n].get_num_of_children() # W - N_0(.)
      prob /= float(self.vocabs.get_num_of_words() * self.ngrams_root[n].get_freq()) # W * N

      w_node = self.ngrams_root[n].get_ngram_last_node([w])
      if w_node is not None:
        w_freq = w_node.get_freq()
        prob += max(float(w_freq - self.b[0]) / self.ngrams_root[n].get_freq(), 0.0)

      return prob

    h_node = self.ngrams_root[n].get_ngram_last_node(h)

    # history is not found so backoff
    if h_node is None:
      return self.compute_prob(w, h[1:], n)

    prob = self.b[len(h)]
    prob *= float(h_node.get_num_of_children()) / h_node.get_freq() # (W - N_0(v,.))/N(v)
    prob *= self.compute_prob(w, h[1:], n) # recursively backoff

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
      bigram_probs += self.compute_prob(w, [10], n=2) # any word for history
      unigram_probs += self.compute_prob(w, [], n=2)

    print('bigram_probs: {}, unigram_probs: {}'.format(bigram_probs, unigram_probs))
    return abs(1.0 - bigram_probs) <= 1e-02 and abs(1.0 - unigram_probs) <= 1e-02

  ########################### Ex5 ##################################

  def perplexity(self, corpus_file, n):
    """Compute the perplexity of the language model on the given corpus
       using n-grams

    :param corpus_file: A string, the path of the corpus
    :param n: An integer, the rank of the grams to be used for computing PP
    :return: A float, perplexity of the LM
    """

    print('Computing model perplexity...')

    LL = 0.0
    norm = 0
    with open(corpus_file, 'r') as test_corpus:
      for line in test_corpus:
        sent = line.strip().split()
        h = [self.corpus.get_start_id()]
        for wrd in sent:
          w = self.corpus.get_idx_by_wrd(wrd)
          if len(h) == n-1:
            prob = self.compute_prob(w, h, n)
            LL += np.log(prob)
            h.append(w)
            h = h[1:]
          else:
            h.append(w)
        LL += np.log(self.compute_prob(self.corpus.get_end_id(), h, n))
        norm += len(sent)+1
    return np.exp(-LL/norm)

#######################################################################

if __name__ == '__main__':
  lm = LM(vocabs_file=vocabulary_path)
  print(lm.get_top_10_ngram_freq(3, lm.vocabs))
  if lm.verify_normalization():
    print('Probabilities are normalized!')
  else:
    print('Probabilities are not normalized!')
  print('Test PP: {}'.format(lm.perplexity(corpus_file=data_test, n=2)))