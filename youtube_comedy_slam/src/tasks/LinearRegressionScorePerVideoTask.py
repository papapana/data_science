import luigi


class LinearRegressionScorePerVideoTask(luigi.Task):
    param = luigi.Parameter(default=42)

    def requires(self):
        return SomeOtherTask(self.param)

    def run(self):
        # Create linear regression object
        regr = linear_model.LinearRegression(n_jobs=-1)

        # Train the model using the training sets
        regr.fit(scipy_train.transpose(), label)

    def output(self):
        return luigi.LocalTarget('/tmp/foo/bar-%s.txt' % self.param)



if __name__ == '__main__':
    luigi.run()