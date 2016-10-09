#!/usr/bin/env sh

NEWFILE="dataset_movies.${1}_${2}_full.csv"
mv dataset_movies.csv $NEWFILE
PREV=`expr $1 - 1`
cat "dataset_movies.1_${PREV}_full.csv" "${NEWFILE}" > "dataset_movies.1_${2}_full.csv"