import os

import gensim
import luigi
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from scipy import io
from sklearn import linear_model
from sklearn.externals import joblib

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
                'test': luigi.LocalTarget(root_path + 'data/processed/label_test_en_per_video.p')}

    def run(self):
        # TODO: Produce output
        """
        CreateLabelsTask: 
        """


class LinearRegressionPerformance(luigi.Task):

    def requires(self):
        """
         This task requires the Linear Regression Model
        """
        return [CreateCorpusTask(), CreateLabelsTask(), CreateDictionaryTask(), LinearRegressionTask()]

    def output(self):
        """
        When this Task is complete, where will it produce output?
        Luigi will check whether this output (specified as a Target)
        exists to determine whether the Task needs to run at all.
        """
        return luigi.LocalTarget(root_path + 'data/processed/lr_performance.log')

    def run(self):
        """
        LinearRegressionPerformance:
        """
        # regr = joblib.load(self.input()[3])
        # id2word = gensim.corpora.Dictionary.load(self.input()[2])
        # scipy_train = io.mmread(self.input()[1]['train']).tocsr()
        # scipy_test = io.mmread(self.input()[1]['test']).tocsr()
        # y_true = label_test
        # y_pred = regr.predict(scipy_test.transpose())
        # r2 = r2_score(y_true, y_pred)
        # mean_sq_error = sqrt(mean_squared_error(y_true, y_pred))
        # print('Variance score: %.2f' % regr.score(scipy_train.transpose(), label))
        # print(r2, mean_sq_error)



if __name__ == '__main__':
    luigi.run()
