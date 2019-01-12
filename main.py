from collections import defaultdict
from index_map import IndexMap
import utils
from trie import Trie

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
    self.oov = 0 # out-of-vocabulary rate

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
                           ylabel='Frequency')

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
    unk_cnt = 0
    unk_id = vocabulary.get_unk_id()
    with open(data_train, 'r') as read_corpus:
      for line in read_corpus:
        sent = line.strip().split(' ')
        tokens.append(start_id)
        for wrd in sent:
          idx = vocabulary.get_idx_by_wrd(wrd)
          if idx == unk_id:
            unk_cnt += 1
          tokens.append(idx)
        tokens.append(end_id)
    return tokens, unk_cnt

  def generate_ngrams(self, n, vocabulary):
    """Generate the ngrams of the corpus and store them in a Trie data structure

    :param n: An integer, the rank of the gram
    :param vocabulary: An IndexMap, either corpus or vocabs
    :return: A Trie representing the ngrams
    """

    print('Reading word tokens from %s' % 'corpus' if vocabulary == self.corpus else 'vocabulary')
    tokens, unk_cnt = self.get_corpus_tokens(vocabulary)
    if vocabulary != self.corpus:
      self.oov = (unk_cnt / self.running_wrds_num) * 100
      print('OOV rate: %.02f %%' % self.oov)

    print('Generating %d-grams from %s' % (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    ngrams = Trie() # root
    for i in range(len(tokens)-n+1):
      ngram = tokens[i:i+n]
      ngrams.add_ngram(ngram)
    return ngrams

  def extract_ngrams_and_freq(self, n, vocabulary):
    """Extract ngrams and their frequencies, display top 10 frequent ngrams,
      plot the count of counts distribution

    :param n: An integer, the rank of the gram
    :param vocabulary: An indexMap, either corpus or vocabs
    """

    root = self.generate_ngrams(n, vocabulary)

    print('Extracting %d-grams using %s' % (n, 'corpus' if vocabulary == self.corpus else 'vocabulary'))

    # file_path = str(n)
    # if vocabulary == self.corpus:
    #   file_path += '-gram.count.corpus.txt'
    # else:
    #   file_path += '-gram.count.vocabs.txt'

    res = root.bfs(n, vocabulary)

    top_10_ngrams = utils.get_top_k_freq_items(res, k=10)
    print('Top 10 %d-grams:' % n, top_10_ngrams)

    print('Preparing to plot count of counts distribution')
    utils.plot_count_of_counts(res, n)

  ##########################################################################

if __name__ == '__main__':
  lm = LM(vocabs_file=vocabulary_path)
  lm.extract_ngrams_and_freq(n=3, vocabulary=lm.corpus)
  # lm.extract_ngrams_and_freq(n=3, vocabulary=lm.vocabs)