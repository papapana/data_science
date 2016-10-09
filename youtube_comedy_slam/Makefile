.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	pip3 install -q -r requirements.txt

data: requirements
# First clean the dataset (training) and then (testing)
# My results (training) : Dataset records before: 912969 Dataset records now: 378581
# My results (testing): Dataset records before: 912969 Dataset records now: 378581
	echo Clean the training dataset...
	src/data/make_dataset.py clean_google_dataset --old-file data/external/comedy_comparisons.train --new-file \
	data/processed/comedy_comparisons.train --full-dataset data/processed/train_dataset_movies.csv
	echo "Clean the testing dataset..."
	src/data/make_dataset.py clean_google_dataset --old-file data/external/comedy_comparisons.test --new-file \
	data/processed/comedy_comparisons.test --full-dataset data/processed/train_dataset_movies.csv

features:

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################