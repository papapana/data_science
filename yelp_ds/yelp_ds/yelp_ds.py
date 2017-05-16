import gensim
import csv
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import json
import random
import linecache
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument


class YelpCorpus(object):
    """
    Iterate over sentences from 
    the Yelp corpus.
    """

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, 'r') as fin:
            yelpreader = csv.DictReader(fin)
            for row in yelpreader:
                #yield row['text'].split()
                # yield row
                yield TaggedDocument(row['text'].split(), [int(row['funny'])])


class CorpusStream:
    def __init__(self, X, prefix):
        self.X = X
        self.prefix = prefix

    def __iter__(self):
        for i, x in enumerate(self.X):
            if hasattr(x, 'split'):
                yield TaggedDocument(x.split(), [self.prefix + '_{0}'.format(i)])
            else:
                yield TaggedDocument('', [self.prefix + '_{0}'.format(i)])


class CorpusDictBowStream:
    def __init__(self, X):
        self.X = X

    def __iter__(self):
        for x in self.X:
            if hasattr(x, 'split'):
                yield x.lower().split()
            else:
                yield ''


class CorpusBowStream:
    def __init__(self, X, dictionary):
        self.X = X
        self.dictionary = dictionary

    def __iter__(self):
        for x in self.X:
            if hasattr(x, 'split'):
                yield self.dictionary.doc2bow(x.split())
            else:
                yield ''


def get_bow_corpus(X, dictionary, pbar=None):
    corpus = []
    for i, x in enumerate(X):
        if not (hasattr(x, 'split')):
            continue
        curbow = dictionary.doc2bow(x.split())
        corpus.append(curbow)
        if not pbar is None:
            pbar.update(1)
    #corpus['score'] = get_label(df, score, dictionary)
    return corpus
                

def build_dataset(fname, model):
    df = pd.read_csv(fname)
    labels = df.funny
    labels.to_pickle('labels.p')
    train = df.text.apply(lambda x: [model[l] for l in x.split() if l in model])
    train.to_pickle('dataset.p')


def build_dataset_streaming(corpus_iter, model, output_file='dataset_no_labels.json'):
    print('Finding dataset length...')
    length = 0
    for _ in tqdm(corpus_iter):
        length += 1

    pbar = tqdm(range(length), desc='Creating dataset...')

    with open(output_file, 'w+') as out:
        for row in corpus_iter:
            out_row = [[model[l].tolist()] for l in row if l in model]
            json.dump(out_row, out)
            pbar.update(1)


def build_dataset_streaming_subset(corpus_iter, model, num_of_sentences=10000, seed=577,
                                   output_file='dataset_no_labels.json'):
    random.seed(seed)
    print('Finding dataset length...')
    length = 4153150
    #   length = 0
    #   for _ in tqdm(corpus_iter):
    #        length += 1

    pbar = tqdm(range(num_of_sentences), desc='Creating subset of dataset...')

    sample_ind = random.sample(range(num_of_sentences), num_of_sentences)
    ind = 0
    with open(output_file, 'w+') as out:
        for index, row in enumerate(corpus_iter):
            if index == sample_ind[ind]:
                out_row = [model[l].tolist() for l in row if l in model]
                json.dump(out_row, out)
                pbar.update(1)
                if ind >= num_of_sentences:
                    break
                ind += 1


class Models:

    def __init__(self, dataset, labels):
        """
        
        :param dataset: pickled df
        :param labels: pickled df
        """
        self.dataset = dataset
        self.labels = labels

    def logistic_regression():
        model = LogisticRegression()
        model.fit(self.dataset, self.labels)


#for epoch in range(10):
#    print('EPOCH: {}'.format(epoch))
#    model.train(np.random.shuffle(vals)
#model.save('full_model_10.mod')

# In [1]: from yelp_ds import *
#
# In [2]: yelp = YelpCorpus('data/raw/yelp_academic_dataset_review.csv')
#
# In [3]: yelp = YelpCorpus('../data/raw/yelp_academic_dataset_review.csv')
#
# In [4]: from gensim.models.word2vec import Word2Vec
#
# In [5]: model = Word2Vec(yelp, size=200, workers=6)
#
# In [6]: model.save('complete_model.m')


# df = pd.read_csv('../data/raw/yelp_academic_dataset_review.csv')
# subset_df = df.sample(n=10000, random_state=577)
# subset_df.to_csv('../data/processed/yelp_subset.csv')


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

