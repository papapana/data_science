import csv
import os
import pickle
from math import sqrt

import gensim
import luigi
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from scipy import io
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error

from src.features.build_features import feature_sum_score
from src.features.build_features import get_corpus_by_video_id

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
root_path = os.environ.get("ROOT_YOUTUBE_PATH")


class InitialProcessedYoutubeDatasets(luigi.ExternalTask):
    def output(self):
        """
        Do initial youtube datasets exist?
        :return: bool
        """
        return {'train': luigi.LocalTarget(
            root_path + 'data/processed/comedy_comparisons.train'),
                'test': luigi.LocalTarget(
            root_path + 'data/processed/comedy_comparisons.test')}


class TimeTaskMixin(object):
    """
    A mixin that when added to a luigi task, will print out
    the tasks execution time to standard out, when the task is
    finished
    """
    @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
    def print_execution_time(self, processing_time):
        print('### PROCESSING TIME ###: ' + str(processing_time))


class ExperimentResults(object):
    def __init__(self):
        self.description = ""
        self.pb_mean_sq = 0.0
        self.pb_variance = 0.0
        self.pa_conf_matrix = [[0, 0], [0, 0]]
        self.pa_conf_matrix_norm = [[0, 0], [0, 0]]
        self.comments = ""
        self.pa_precision = 0.0
        self.pa_recall = 0.0
        self.pa_f1_score = 0.0
        self.pb_r2_score = 0.0
        self.write_mode = 'w+'
        self.filename = 'temp'

    def set_results(self, pb_mean_sq, pb_r2_score, pb_variance, pa_conf_matrix, comments=""):
        self.pb_mean_sq = pb_mean_sq
        self.pb_variance = pb_variance
        self.pa_conf_matrix = pa_conf_matrix
        self.pb_r2_score = pb_r2_score
        self.comments = comments
        # [[tp, fn], [fp, tn]]
        tp = self.pa_conf_matrix[0][0]
        fn = self.pa_conf_matrix[0][1]
        fp = self.pa_conf_matrix[1][0]
        tn = self.pa_conf_matrix[1][1]
        self.pa_precision = tp / (tp + fp)
        self.pa_recall = tp / (tp + fn)
        self.pa_f1_score = 2 * (self.pa_precision * self.pa_recall) / (self.pa_precision + self.pa_recall)
        self.pa_conf_matrix_norm = [[tp / (tp + fn), fn / (tp + fn)], [fp / (fp + tn), tn / (fp + tn)]]

    def get_experiment_result(self):
        return [self.description, self.pb_mean_sq, self.pb_r2_score, self.pb_variance, self.pa_precision, self.pa_recall,
                self.pa_f1_score, self.pa_conf_matrix, self.pa_conf_matrix_norm, self.comments]

    def write_experiment_result(self):
        with open(self.filename, self.write_mode) as fout:
            wr = csv.writer(fout)
            wr.writerow(self.get_experiment_result())


class AddFeatureSumScoreTask(luigi.Task, TimeTaskMixin):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to 
        compute the task dependency graph.
        """
        return InitialProcessedYoutubeDatasets()

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target) 
        exists to determine whether the Task needs to run at all.
        """
        return {'train': luigi.LocalTarget(
            root_path + 'data/processed/comedy_videos_with_sum_feature.train'),
                'test':  luigi.LocalTarget(
            root_path + 'data/processed/comedy_videos_with_sum_feature.test')}

    def run(self):
        """
        How do I run this Task?
        Luigi will call this method if the Task needs to be run.
        """
        # We can do anything we want in here, from calling python
        # methods to running shell scripts to calling APIs
        train = pd.read_csv(root_path + 'data/processed/comedy_comparisons.train')
        test = pd.read_csv(root_path + 'data/processed/comedy_comparisons.test')
        score_train_not_norm = feature_sum_score(train, normalized=False)
        score_test_not_norm = feature_sum_score(test, normalized=False)
        score_train_not_norm.to_csv(root_path + 'data/processed/comedy_videos_with_sum_feature.train')
        score_test_not_norm.to_csv(root_path + 'data/processed/comedy_videos_with_sum_feature.test')


class AddLangFeatureTask(luigi.Task):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to
        compute the task dependency graph.
        """
        # TODO: require the dataset without the language feature
        return []

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return {'train': luigi.LocalTarget(root_path + 'data/processed/train_dataset_with_lang_feature.csv'),
                'test': luigi.LocalTarget(root_path + 'data/processed/test_dataset_with_lang_feature.csv')}

    def run(self):
        """
        AddLangFeatureTask :
        """
        pass


class CreateDictionaryTask(luigi.Task):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to
        compute the task dependency graph.
        """
        return []

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return luigi.LocalTarget(root_path + 'data/processed/youtube_comments_en.dict')

    def run(self):
        """
        CreateDictionaryTask:
        """
        pass


class CreateTrainTestIndicesTask(luigi.Task):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to
        compute the task dependency graph.
        """
        return [AddLangFeatureTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return {'indices': {'train': luigi.LocalTarget(root_path + 'data/processed/video_train_indices.p'),
                            'test': luigi.LocalTarget(root_path + 'data/processed/video_test_indices.p')},
                'video_id': {'train': luigi.LocalTarget(root_path + 'data/processed/video_id_train.p'),
                             'test': luigi.LocalTarget(root_path + 'data/processed/video_id_test.p')}
                }

    def run(self):
        """
        CreateTrainTestIndicesTask:
        """
        pass


class CreateCorpusTask(luigi.Task, TimeTaskMixin):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to
        compute the task dependency graph.
        """
        return [AddLangFeatureTask(), CreateDictionaryTask(), CreateTrainTestIndicesTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return {'train': luigi.LocalTarget(root_path + 'data/processed/scipy_corpus_train_en_per_video_id.mtx'),
                'test': luigi.LocalTarget(root_path + 'data/processed/scipy_corpus_test_en_per_video_id.mtx')}

    def run(self):
        """
        CreateCorpusTask:
        Create the full test and training corpus
        """
        all_data = pd.concat([self.output()[0]['train'], self.output()[0]['test']])
        corpus = get_corpus_by_video_id(all_data, id2word)
        scipy_corpus = gensim.matutils.corpus2csc(corpus)
        scipy_corpus_train = scipy_corpus[:, train_indices]
        scipy_corpus_test = scipy_corpus[:, test_indices]
        io.mmwrite(self.output()['train'], scipy_corpus_train)
        io.mmwrite(self.output()['test'], scipy_corpus_test)


class LinearRegressionTask(luigi.Task, TimeTaskMixin):

    def requires(self):
        """
        LR needs score and corpus (for features)
        """
        return [AddFeatureSumScoreTask(), CreateCorpusTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target) 
        exists to determine whether the Task needs to run at all.
        """
        return luigi.LocalTarget(root_path + 'models/linear_regression_en.pkl')

    def run(self):
        """
        LinearRegressionTask:
        """
        scipy_train = io.mmread(self.input()[1]['train']).tocsr()
        # Create linear regression object
        regr = linear_model.LinearRegression(n_jobs=-1)
        # Train the model using the training sets
        regr.fit(scipy_train.transpose(), label)
        joblib.dump(regr, self.output())


class RidgeRegressionTask(luigi.Task, TimeTaskMixin):

    def requires(self):
        """
        LR needs score and corpus (for features)
        """
        return [AddFeatureSumScoreTask(), CreateCorpusTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return luigi.LocalTarget(root_path + 'models/ridge_regression_en.pkl')

    def run(self):
        """
        RidgeRegressionTask:
        """
        scipy_train = io.mmread(self.input()[1]['train']).tocsr()
        # Create linear regression object
        regr = linear_model.RidgeCV(n_jobs=-1)
        # Train the model using the training sets
        regr.fit(scipy_train.transpose(), label)
        joblib.dump(regr, self.output())


class CreateLabelsTask(luigi.Task, TimeTaskMixin):

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to 
        compute the task dependency graph.
        """
        return []

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target) 
        exists to determine whether the Task needs to run at all.
        """
        return {'train': luigi.LocalTarget(root_path + 'data/processed/label_en_per_video_id.p'),
                'test': luigi.LocalTarget(root_path + 'data/processed/label_test_en_per_video_id.p')}

    def run(self):
        # TODO: Produce output
        """
        CreateLabelsTask: 
        """
        pass


class LinearRegressionPrediction(luigi.Task):
    # Example parameter for our task: a
    # date for which a report should be run
    # report_date = luigi.DateParameter()

    def requires(self):
        """
        Which other Tasks need to be complete before
        this Task can start? Luigi will use this to
        compute the task dependency graph.
        """
        return [CreateCorpusTask(), LinearRegressionTask(), CreateLabelsTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return {'train': luigi.LocalTarget(root_path + 'data/processed/linear_predictions_en_per_video_train.p'),
                'test': luigi.LocalTarget(root_path + 'data/processed/linear_predictions_en_per_video_test.p'),
                'variance': luigi.LocalTarget(root_path + 'data/processed/linear_predictions_en_per_video_variance.p')}

    def run(self):
        """
        LinearRegressionPrediction:
        Predict for both the training and the testing dataset
        """
        regr = joblib.load(self.input()[1].path)
        scipy_test = io.mmread(self.input()[0]['test'].path).tocsr()
        y_pred = regr.predict(scipy_test.transpose())
        joblib.dump(y_pred, self.output()['test'].path)
        scipy_train = io.mmread(self.input()[0]['train'].path).tocsr()
        y_pred = regr.predict(scipy_train.transpose())
        joblib.dump(y_pred, self.output()['train'].path)
        label_train_path = self.input()[2]['train'].path
        label_train = pickle.load(open(label_train_path, 'rb'))
        variance = regr.score(scipy_train.transpose(), label_train)
        joblib.dump(variance, self.output()['variance'].path)


class LinearRegressionPerformance(luigi.Task):

    def __init__(self):
        luigi.Task.__init__(self)
        self.test_results = ExperimentResults()
        self.train_results = ExperimentResults()

    def requires(self):
        """
         This task requires the Linear Regression Model
        """
        return [LinearRegressionPrediction(), CreateLabelsTask(), CreateDictionaryTask(),
                AddFeatureSumScoreTask(), InitialProcessedYoutubeDatasets()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return luigi.LocalTarget(root_path + 'data/processed/lr_performance.csv')

    def run(self):
        """
        LinearRegressionPerformance:
        """
        label_test_path = self.input()[1]['test'].path

        label_test = pickle.load(open(label_test_path, 'rb'))
        y_true = label_test
        y_pred = joblib.load(self.input()[0]['test'].path)
        r2 = r2_score(y_true, y_pred)
        mean_sq_error = sqrt(mean_squared_error(y_true, y_pred))

        video_score_test = pd.read_csv(self.input()[3]['test'].path, header=None, index_col=0)[1].to_dict()
        initial_dataset_test = pd.read_csv(self.input()[4]['test'].path)

        # Transform to problem A
        bin_y_true = self.predict_initial_problem(initial_dataset_test, video_score_test)
        trans_y_pred = dict(zip(video_score_test.keys(), y_pred))
        bin_y_pred = self.predict_initial_problem(initial_dataset_test, trans_y_pred)
        cnf_matrix = confusion_matrix(bin_y_true, bin_y_pred)
        self.filename = self.output().path
        self.description = "LR, English, score per video, training set, score not normalized"
        pb_variance = joblib.load(self.input()[0]['variance'].path)
        self.set_results(mean_sq_error, r2, pb_variance, cnf_matrix)
        self.write_experiment_result()

    @staticmethod
    def predict_initial_problem(initial_dataset, video_score_test):
        bin_ytest = []
        for row in initial_dataset.itertuples():
            pred1 = video_score_test[row.video_id1] if row.video_id1 in video_score_test else 0.0
            pred2 = video_score_test[row.video_id2] if row.video_id2 in video_score_test else 0.0
            bin_ytest += [0 if pred1 >= pred2 else 1]
        return bin_ytest


if __name__ == '__main__':
    luigi.run()
