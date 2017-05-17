import datetime
import os
import pickle
from collections import OrderedDict
from collections import defaultdict
# for timing
from contextlib import contextmanager
from random import shuffle
from timeit import default_timer

import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from scipy import io
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from .yelp_ds import *

x_train_str = 'data/processed/x_train_{0}.hdf'
x_test_str = 'data/processed/x_test_{0}.hdf'
y_train_str = 'data/processed/y_train_{0}.hdf'
y_test_str = 'data/processed/y_test_{0}.hdf'
vocab_model = 'models/mod_voc_{0}.m'


def create_train_test_datasets(tag, cores=6):
    files = [x_train_str.format(tag), x_test_str.format(tag), y_train_str.format(tag), y_test_str.format(tag)]
    if not all(os.path.exists(file) for file in files):
        print('Creating train and test datasets, please wait...')
        if tag == 'small':
            yelp_df = pd.read_csv('data/processed/yelp_subset_with_flag.csv')
        else:
            yelp_df = pd.read_csv('data/processed/yelp_big.csv')
        x = yelp_df.text
        y = yelp_df.funny_flag
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train.to_hdf(files[0], x_train_str.format(tag))
        x_test.to_hdf(files[1],  x_test_str.format(tag))
        y_train.to_hdf(files[2], y_train_str.format(tag))
        y_test.to_hdf(files[3],  y_test_str.format(tag))
    else:
        x_train = pd.read_hdf(x_train_str.format(tag))
        x_test = pd.read_hdf(x_test_str.format(tag))
        y_train = pd.read_hdf(y_train_str.format(tag))
        y_test = pd.read_hdf(y_test_str.format(tag))
        print('Train and test datasets in place.')
    return x_train, x_test, y_train, y_test


def get_models_d2vec(cores=6):
    """
    Configure models and return them
    :param cores: number of cores to run on
    :return: the models
    """
    return [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window sizmae
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]


def build_vocab_d2vec(tag, cores=6):
    """
    Build vocabulary for dataset
    :param tag: 
    :param cores: 
    :return: 
    """
    d2vec = Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores)
    file = x_train_str.format(tag)
    if os.path.exists(file):
        x_train = pd.read_hdf(file)
        mod_file = vocab_model.format(tag)
        if not os.path.exists(mod_file):
            print('Building vocabulary...')
            d2vec.build_vocab(CorpusStream(x_train, 'train'))
            print('Vocabulary built')
        else:
            print('Vocabulary exists. Loading...')
            d2vec.load(mod_file)
            print('Vocabulary loaded')
        return d2vec


def create_bow_tfidf_corpuses(tag):
    file = x_train_str.format(tag)
    x_train = pd.read_hdf(file)
    dict_file = 'data/processed/dict_train_{0}.dict'
    if os.path.exists(dict_file):
        print('Dictionary exists, loading...')
        dict_train = gensim.corpora.Dictionary().load(dict_file)
    else:
        print('Building dictionary...')
        corpus_dict = CorpusDictBowStream(x_train)
        dict_train = gensim.corpora.Dictionary(corpus_dict)
        dict_train.save(dict_file)

    # Create MMCorpus for bow
    mm_corpus_bow_file = 'data/processed/mm_corpus_{0}.mm'.format(tag)
    if os.path.exists(mm_corpus_bow_file):
        print('Bag of words corpus exists, loading...')
        mm_corpus_bow = gensim.corpora.MmCorpus(mm_corpus_bow_file)
    else:
        print('Creating bag of words corpus...')
        corpus_stream = CorpusBowStream(x_train, dict_train)
        gensim.corpora.MmCorpus.serialize(mm_corpus_bow_file, corpus_stream)
        mm_corpus_bow = gensim.corpora.MmCorpus(mm_corpus_bow_file)

    mm_corpus_tfidf_file = 'data/processed/mm_corpus_tfidf_{0}.mm'.format(tag)

    if os.path.exists(mm_corpus_tfidf_file):
        print('Tf-Idf corpus exists, loading...')
        mm_corpus_tfidf = gensim.corpora.MmCorpus(mm_corpus_tfidf_file)
    else:
        print('Creating tfidf corpus...')
        corpus_stream = gensim.models.TfidfModel(mm_corpus_bow)
        gensim.corpora.MmCorpus.serialize(mm_corpus_tfidf_file, corpus_stream[mm_corpus_bow])
        mm_corpus_tfidf = gensim.corpora.MmCorpus(mm_corpus_tfidf_file)

    return mm_corpus_bow, mm_corpus_tfidf


def train_logistic_regression_bow(tag, cores=6):
    file = y_train_str.format(tag)
    y_train = pd.read_hdf(file)
    mm_corpus_bow, _ = create_bow_tfidf_corpuses(tag)
    model_file = 'models/logit_bow_{0}'.format(tag)
    if os.path.exists(model_file):
        print('Logistic regression for bow exists, loading...')
        logit_bow = joblib.load(model_file)
    else:
        # Making the corpus sparse
        print('Training Logistic Regression for bow...')
        corpus_bow_sparse = gensim.matutils.corpus2csc(mm_corpus_bow)
        corpus_bow_sparse = np.nan_to_num(corpus_bow_sparse)
        # ll = [i for i, row in enumerate(corpus_bow_sparse.data) if not np.all(np.isnan(row))]
        # # Remove NaN
        # print('- Removing NaN, corpus_bow shape before {0}'.format(corpus_bow_sparse.shape))
        # corpus_bow_sparse = corpus_bow_sparse[ll, :]
        corpus_bow_sparse = corpus_bow_sparse.transpose()
        # print('After removing NaN, corpus_bow shape {0}'.format(corpus_bow_sparse.shape))
        # print('Removing also y_train nan')
        # y_train = np.array(y_train)
        # y_train = y_train[ll]
        # print('y_train shape after keeping all not nan: {0}'.format(len(y_train)))
        logit_bow = LogisticRegression(n_jobs=cores)
        logit_bow.fit(corpus_bow_sparse, y_train)
        joblib.dump(logit_bow, model_file)
    return logit_bow


def train_logistic_regression_tfidf(tag, cores=6):
    file = y_train_str.format(tag)
    y_train = pd.read_hdf(file)
    _, mm_corpus_tfidf = create_bow_tfidf_corpuses(tag)
    model_file = 'models/logit_tfidf_{0}'.format(tag)
    if os.path.exists(model_file):
        print('Logistic Regression model for tfidf exists, loading...')
        logit_tfidf = joblib.load(model_file)
    else:
        # Making the corpus sparse
        print('Logistic Regression model for tfidf is being built...')
        corpus_tfidf_sparse = gensim.matutils.corpus2csc(mm_corpus_tfidf)
        corpus_tfidf_sparse = np.nan_to_num(corpus_tfidf_sparse)
        # ll = [i for i, row in enumerate(corpus_tfidf_sparse.data) if not np.all(np.isnan(row))]
        # Remove NaN
        # print('Removing NaN, corpus_bow shape before {0}'.format(corpus_tfidf_sparse.shape))
        # corpus_tfidf_sparse = corpus_tfidf_sparse[ll, :]
        corpus_tfidf_sparse = corpus_tfidf_sparse.transpose()
        # print('After removing NaN, corpus_bow shape {0}'.format(corpus_tfidf_sparse.shape))
        # print('Removing also y_train nan')
        # y_train = np.array(y_train)
        # y_train = y_train[ll]
        # print('y_train shape after keeping all not nan: {0}'.format(len(y_train)))
        logit_tfidf = LogisticRegression(n_jobs=cores)
        logit_tfidf.fit(corpus_tfidf_sparse, y_train)
        joblib.dump(logit_tfidf, model_file)
    return logit_tfidf


def get_test_corpus_bow(tag):
    file = x_test_str.format(tag)
    x_test = pd.read_hdf(file)
    corpus_test_file = 'data/processed/corpus_test_{0}.m'.format(tag)
    if os.path.exists(corpus_test_file):
        mm_corpus_test_bow = gensim.corpora.MmCorpus(corpus_test_file)
    else:
        dict_file = 'data/processed/dict_train_{0}.dict'
        dict_train = gensim.corpora.Dictionary().load(dict_file)
        corpus_stream_test = CorpusBowStream(x_test, dict_train)
        gensim.corpora.MmCorpus.serialize(corpus_test_file, corpus_stream_test)
        mm_corpus_test_bow = gensim.corpora.MmCorpus(corpus_test_file)
    return mm_corpus_test_bow


def get_test_corpus_tfidf(tag):
    mm_corpus = get_test_corpus_bow(tag)
    corpus_test_file = 'data/processed/corpus_test_tfidf_{0}.m'.format(tag)
    if os.path.exists(corpus_test_file):
        mm_corpus_test_tfidf = gensim.corpora.MmCorpus(corpus_test_file)
    else:
        corpus_test_tfidf = gensim.models.TfidfModel(mm_corpus)
        gensim.corpora.MmCorpus.serialize(corpus_test_file, corpus_test_tfidf[mm_corpus])
        mm_corpus_test_tfidf = gensim.corpora.MmCorpus(corpus_test_file)
    return mm_corpus_test_tfidf


def predict_bow_tfidf(tag, cores=6):
    bow = train_logistic_regression_bow(tag, cores)
    tfidf = train_logistic_regression_tfidf(tag, cores)

    corpuses_test = get_test_corpus_bow(tag)
    corpuses_test_tfidf = get_test_corpus_tfidf(tag)

    file = y_test_str.format(tag)
    y_true = pd.read_hdf(file)

    files = ['data/processed/accuracy_bow_{0}.p'.format(tag),
             'data/processed/confusion_matrix_bow_{0}.p'.format(tag),
             'data/processed/accuracy_tfidf_{0}.p'.format(tag),
             'data/processed/confusion_matrix_tfidf_{0}.p'.format(tag),
             'data/processed/f1_bow_{0}.p'.format(tag),
             'data/processed/f1_tfidf_{0}.p'.format(tag),
             'data/processed/y_pred_bow_{0}.p'.format(tag),
             'data/processed/y_pred_tfidf_{0}.p'.format(tag)
             ]
    if all([os.path.exists(file) for file in files]):
        print('Predictions exist, loading...')
        p_metrics_bow = pickle.load(open(files[0], 'rb'))
        p_metrics_conf_matrix_bow = pickle.load(open(files[1], 'rb'))
        p_metrics_tfidf = pickle.load(open(files[2], 'rb'))
        p_metrics_conf_matrix_tfidf = pickle.load(open(files[3], 'rb'))
        f1_bow = pickle.load(open(files[4], 'rb'))
        f1_tfidf = pickle.load(open(files[5], 'rb'))
        y_pred_bow = pickle.load(open(files[6], 'rb'))
        y_pred_tfidf = pickle.load(open(files[7], 'rb'))
    else:
        print('Making predictions...')
        dict_file = 'data/processed/dict_train_{0}.dict'
        dict_train = gensim.corpora.Dictionary().load(dict_file)
        x_test_sparse = gensim.matutils.corpus2csc(corpuses_test, num_terms=len(dict_train)).transpose()
        y_pred_bow = bow.predict(x_test_sparse)
        p_metrics_bow = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_bow)
        p_metrics_conf_matrix_bow = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_bow)
        pickle.dump(p_metrics_bow, open(files[0], 'wb'))
        pickle.dump(y_pred_bow, open(files[6], 'wb'))
        pickle.dump(f1_bow, open(files[4], 'wb'))
        pickle.dump(p_metrics_conf_matrix_bow, open(files[1], 'wb'))

        x_test_sparse = gensim.matutils.corpus2csc(corpuses_test_tfidf, num_terms=len(dict_train)).transpose()
        y_pred_tfidf = tfidf.predict(x_test_sparse)
        p_metrics_tfidf = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_tfidf)
        p_metrics_conf_matrix_tfidf = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_tfidf)
        pickle.dump(p_metrics_tfidf, open(files[2], 'wb'))
        pickle.dump(f1_tfidf, open(files[5], 'wb'))
        pickle.dump(y_pred_tfidf, open(files[7], 'wb'))
        pickle.dump(p_metrics_conf_matrix_tfidf, open(files[3], 'wb'))

    print('name: {0} - accuracy: {1} '.format('bow', p_metrics_bow))
    print('name: {0} - accuracy: {1} '.format('tfidf', p_metrics_tfidf))
    print('confusion matrix bow:')
    print(p_metrics_conf_matrix_bow)
    print('confusion matrix tfidf: ')
    print(p_metrics_conf_matrix_tfidf)

    return p_metrics_bow, p_metrics_conf_matrix_bow, p_metrics_tfidf, p_metrics_conf_matrix_tfidf, y_pred_bow, \
           y_pred_tfidf


def create_and_train_models_d2vec(tag, cores=6):
    """
    Build vocabulary and train models
    :param tag: small or big 
    :param cores: number of cores
    :return: the current models
    """
    simple_models = get_models_d2vec(cores)
    model_files = get_models_filename_d2vec(tag)
    if all([os.path.exists(file) for file in model_files]):
        print('Models exist, loading...')
        for i, fname in enumerate(model_files):
            simple_models[i] = Doc2Vec.load(fname)
        models_by_name = OrderedDict((str(model), model) for model in simple_models)
        return models_by_name
    else:
        print('Building models...')
        voc_model = build_vocab_d2vec(tag, cores)
        # Share vocabulary between models
        for model in simple_models:
            model.reset_from(voc_model)

        models_by_name = OrderedDict((str(model), model) for model in simple_models)
        print('Training models...')
        print("START %s" % datetime.datetime.now())
        best_error = defaultdict(lambda: 1.0)  # to selectively-print only best errors achieved

        alpha, min_alpha, passes = (0.025, 0.001, 20)
        alpha_delta = (alpha - min_alpha) / passes
        file = x_train_str.format(tag)
        x_train = pd.read_hdf(file)
        train_list = x_train.tolist()

        for epoch in range(passes):
            shuffle(train_list)  # shuffling gets best results

            for name, train_model in models_by_name.items():
                # train
                duration = 'na'
                train_model.alpha, train_model.min_alpha = alpha, alpha
                with elapsed_timer() as elapsed:
                    train_model.train(CorpusStream(train_list, 'train'), total_examples=train_model.corpus_count,
                                      epochs=train_model.iter)
                    duration = '%.1f' % elapsed()

            print('completed pass %i at alpha %f' % (epoch + 1, alpha))
            alpha -= alpha_delta

        print("END %s" % str(datetime.datetime.now()))
        for name, model in models_by_name.items():
            name = name.replace('/', '').replace(',', '_')
            model.save('models/{0}_{1}.m'.format(name, tag))

    return models_by_name


@contextmanager
def elapsed_timer():
    """
    Measure time generator
    :return: 
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def get_models_filename_d2vec(tag):
    """
    Returns modmel's filename according to each tag
    :param tag: small or big
    :return: filename for model
    """
    models = get_models_d2vec()
    models_by_name = OrderedDict((str(model), model) for model in models)
    fnames = [name.replace('/', '').replace(',', '_') for name, _ in models_by_name.items()]
    fnames = ['models/{0}_{1}.m'.format(name, tag) for name in fnames]
    return fnames


def get_fname(name):
    return name.replace('/', '').replace(',', '_')


def create_train_corpuses_d2vec(tag, cores=6):
    """
    Creates the train corpuses
    :param tag: small or big
    :param cores: the number of cores
    :return: the train corpuses
    """
    models = create_and_train_models_d2vec(tag, cores)
    corpuses = OrderedDict((str(model), model) for model in models)

    for i, (name, model) in enumerate(models.items()):
        cur_file = 'data/processed/corpus_{0}_{1}.mtx'.format(i, tag)
        if os.path.exists(cur_file):
            print('Train corpuses exist, loading...')
            corpuses[name] = io.mmread(cur_file)
        else:
            print('Creating training corpuses...')
            corpuses[name] = np.vstack([model.docvecs[d] for d in range(len(model.docvecs))])
            io.mmwrite(cur_file, corpuses[name])
    return corpuses


def create_test_corpuses_d2vec(tag, cores=6):
    """
    Creates the test corpuses
    :param tag: small or big
    :param cores: the number of cores
    :return: the test corpuses
    """
    models = create_and_train_models_d2vec(tag, cores)
    corpuses = OrderedDict((str(model), model) for model in models)
    file = x_test_str.format(tag)
    x_test = pd.read_hdf(file).tolist()
    for i, (name, model) in enumerate(models.items()):
        cur_file = 'data/processed/corpus_test_{0}_{1}.mtx'.format(i, tag)
        if os.path.exists(cur_file):
            print('Test corpuses exist, loading...')
            corpuses[name] = io.mmread(cur_file)
        else:
            print('Creating test corpuses...')
            corpuses[name] = np.vstack([model.infer_vector(x_test[d]) for d in range(len(x_test))])
            io.mmwrite(cur_file, corpuses[name])
    return corpuses


def train_logistic_regression_d2vec(tag, cores=6):
    models = create_and_train_models_d2vec(tag, cores)
    corpuses_train = create_train_corpuses_d2vec(tag, cores)
    lrs = OrderedDict((str(model), model) for model in models)
    file = y_train_str.format(tag)
    y_train = pd.read_hdf(file).tolist()
    for i, (name, corpus) in enumerate(corpuses_train.items()):
        file = 'models/lr_{0}_{1}'.format(get_fname(name), tag)
        if os.path.exists(file):
            print('Logistic Regression model exists, loading...')
            lrs[name] = joblib.load(file)
        else:
            print('Training Logistic Regression...')
            lrs[name] = LogisticRegression(n_jobs=cores)
            lrs[name].fit(corpus, y_train)
            joblib.dump(lrs[name], file)
    return lrs


def predictions_d2vec(tag, cores=6):
    """
    Return confusion matrices and accuracies
    :param tag: 
    :param cores: 
    :return: 
    """
    corpuses_test = create_test_corpuses_d2vec(tag, cores)
    lrs = train_logistic_regression_d2vec(tag, cores)
    file = y_test_str.format(tag)
    y_test = pd.read_hdf(file).tolist()
    models = create_and_train_models_d2vec(tag, cores)
    p_metrics = OrderedDict((str(model), model) for model in models)
    p_metrics_f1 = OrderedDict((str(model), model) for model in models)
    y_preds = OrderedDict((str(model), model) for model in models)
    p_metrics_conf_matrix = OrderedDict((str(model), model) for model in models)
    for i, (name, model) in enumerate(lrs.items()):
        files = ['data/processed/accuracy_{0}_{1}.p'.format(get_fname(name), tag),
                 'data/processed/confusion_matrix_{0}_{1}.p'.format(get_fname(name), tag),
                 'data/processed/f1_{0}_{1}.p'.format(get_fname(name), tag),
                 'data/processed/y_preds_{0}_{1}.p'.format(get_fname(name), tag)]
        if all([os.path.exists(file) for file in files]):
            print('Predictions exist, loading...')
            p_metrics[name] = pickle.load(open(files[0], 'rb'))
            p_metrics_conf_matrix[name] = pickle.load(open(files[1], 'rb'))
            y_preds[name] = pickle.load(open(files[3], 'rb'))
        else:
            print('Making predictions...')
            y_preds[name] = model.predict(corpuses_test[name])
            p_metrics[name] = metrics.accuracy_score(y_true=y_test, y_pred=y_preds[name])
            p_metrics_conf_matrix[name] = metrics.confusion_matrix(y_true=y_test, y_pred=y_preds[name])
            pickle.dump(p_metrics[name], open(files[0], 'wb'))
            pickle.dump(p_metrics_f1[name], open(files[2], 'wb'))
            pickle.dump(p_metrics_conf_matrix[name], open(files[1], 'wb'))
            pickle.dump(y_preds[name], open(files[3], 'wb'))
        print('name: {0} - accuracy: {1} - f1: {2}'.format(get_fname(name), p_metrics[name], p_metrics_f1))
        print('name: {0} - confusion matrix:\n{1}'.format(get_fname(name), p_metrics_conf_matrix[name]))
    return p_metrics_conf_matrix, p_metrics, y_preds


def main(tag):
    create_train_test_datasets(tag)
    predict_bow_tfidf(tag)
    predictions_d2vec(tag)
