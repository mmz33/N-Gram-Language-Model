from collections import defaultdict
from index_map import IndexMap
import utils
from trie import Trie

# data paths
data_train = 'data/train.corpus'
data_test = 'data/test.corpus'
vocabulary = 'data/vocabulary'

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

  def prepare_lm(self):
    """Prepare the language model for analysis and computations"""

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

    utils.plot_iterable(sorted(self.sent_len_to_freq.items(), key=lambda kv: kv[0]),
                        xlabel='Sentence length',
                        ylabel='Frequency',
                        log10_space=True)

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

    return utils.get_top_k_freq_words(lm.get_word_freq_dict(), k=10)

  ############################# Ex2 (b) ####################################

  def get_corpus_tokens(self, with_unk=False):
    """Read corpus and store the tokens in a list

    If with_unk is True, then unk_cnt is used to store the number
    of OOV words to compute it's rate later

    :return: A list of word tokens with start and end symbols
    """

    tokens = []
    start_id = self.corpus.get_start_id()
    end_id = self.corpus.get_end_id()
    unk_cnt = 0
    unk_id = self.corpus.get_unk_id()
    with open(data_train, 'r') as read_corpus:
      for line in read_corpus:
        sent = line.strip().split(' ')
        tokens.append(start_id)
        for wrd in sent:
          idx = self.corpus.get_idx_by_wrd(wrd)
          if with_unk and idx not in self.vocabs:
            tokens.append(unk_id)
            unk_cnt += 1
          else:
            tokens.append(idx)
        tokens.append(end_id)
    return tokens, unk_cnt

  def generate_ngrams(self, n):
    """Generate the ngrams of the corpus and store them in a Trie data structure

    :param n: An integer, the rank of the gram
    :return: A Trie representing the ngrams
    """

    tokens = self.get_corpus_tokens()
    ngrams = Trie() # root
    for i in range(len(tokens)-n+1):
      ngram = tokens[i:n]
      ngrams.add_ngram(ngram)
    return ngrams

  ##########################################################################

if __name__ == '__main__':
  lm = LM()