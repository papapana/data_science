#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv


def num_of_lines(file):
    """
    Returns the number of lines of file
    :param file: the filename
    :return: the number of lines
    """
    with open(file, 'r') as f:
        return sum(1 for _ in f)


def clean_google_dataset(full_dataset, google_dataset_old, google_dataset_new):
    """
    Cleans the google_dataset_old of youtube ids that do not exist in the full_dataset
    and writes the result in google_dataset_new
    :param full_dataset: the full downloaded dataset
    :param google_dataset_old: the filename of the google dataset (format id1,id2,(left|right)
    :param google_dataset_new: the filename of the clean dataset
    :return:
    """

    # Reading a big csv results in reading limitations of the field limit in the csv module
    # To circumvent this we make the field limit as big as we can
    # see http://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    maxInt = sys.maxsize
    decrement = True

    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    csv.field_size_limit(maxInt)

    youtube_ids = {}
    with open(full_dataset) as dataset:
        ds = csv.reader(dataset, delimiter=',')
        for row in ds:
            youtube_ids[row[0]] = True

    with open(google_dataset_old) as old_file:
        with open(google_dataset_new, 'w') as new_file:
            old_ds = csv.reader(old_file, delimiter=',')
            new_file.write("video_id1,video_id2,funnier")

            for row in old_ds:
                if row[0] in youtube_ids and row[1] in youtube_ids:
                    new_file.write(",".join(row) + "\n")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    PARSER = argparse.ArgumentParser()
    SUB_PARSER = PARSER.add_subparsers(help='Command to execute', title='actions', dest='action')
    SUB_PARSER.required = True
    SP = SUB_PARSER.add_parser('clean_google_dataset', help='Will clean the published google dataset')
    SP.add_argument('--old-file', type=argparse.FileType('r'), help='Old google dataset', required=True)
    SP.add_argument('--new-file', type=argparse.FileType('w'), help='New google dataset', required=True)
    SP.add_argument('--full-dataset', type=argparse.FileType('r'), help='Full dataset', required=True)
    ARGS = PARSER.parse_args()
    if ARGS.action == 'clean_google_dataset':
        try:
            full_dataset = ARGS.full_dataset.name
            google_dataset_old = ARGS.old_file.name
            google_dataset_new = ARGS.new_file.name
            records_before = num_of_lines(google_dataset_old)
            print("Cleaning dataset of inexisting names...")
            clean_google_dataset(full_dataset, google_dataset_old, google_dataset_new)
            records_new = num_of_lines(google_dataset_new)
            print("Dataset records before:", records_before, "Dataset records now:", records_new)
        except Exception as err:
            logging.error("Unexpected error:", sys.exc_info()[0], "Exception", err)



