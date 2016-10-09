#!/usr/bin/env sh

NEWFILE="test_dataset_movies.${1}_${2}_full.csv"
mv test_dataset_movies.csv $NEWFILE
PREV=`expr $1 - 1`
cat "test_dataset_movies.1_${PREV}_full.csv" "${NEWFILE}" > "test_dataset_movies.1_${2}_full.csv"