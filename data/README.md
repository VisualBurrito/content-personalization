# Data

Raw data files are not committed to this repo.

## MovieLens 32M
Download from: https://files.grouplens.org/datasets/movielens/ml-32m.zip
Unzip into: data/raw/movielens/

## TMDB Movies Dataset
Download from: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
Place these files into: data/raw/tmdb/
  - movies_metadata.csv
  - links.csv
  - credits.csv
  - keywords.csv

## Automated download
Run: ./scripts/setup_data.sh full
Requires: wget, unzip, kaggle CLI configured
