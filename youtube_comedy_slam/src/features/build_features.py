#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import argparse
import pandas as pd


def feature_sum_score(dataset_df):
    """
    This functions creates a score for each video which is the sum(better)-sum(worse)
    :param dataset_df: The dataset panda dataframe
    :return: Normalized series with the final score [-1..1] and the video_id's as index
    """
    better1 = dataset_df[dataset_df.funnier == 'left'].video_id1.sort_index().value_counts()
    better2 = dataset_df[dataset_df.funnier == 'right'].video_id2.sort_index().value_counts()
    better = better1.add(better2, fill_value=0).sort_index()
    worse1 = dataset_df[dataset_df.funnier == 'right'].video_id1.sort_index().value_counts()
    worse2 = dataset_df[dataset_df.funnier == 'left'].video_id2.sort_index().value_counts()
    worse = worse1.add(worse2, fill_value=0).sort_index()
    score = better.subtract(worse, fill_value=0).sort_index()
    max_score = score.abs().max()
    assert max_score != 0
    norm_score = score / max_score

    return norm_score


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
