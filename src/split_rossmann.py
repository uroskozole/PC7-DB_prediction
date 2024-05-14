import numpy as np
import pandas as pd

from pathlib import Path

from utils.metadata import Metadata
from utils.data import load_tables, remove_sdv_columns

DATA_DIR = "./data"

TRAIN_START_DATE = pd.to_datetime('2013-01-01')
TRAIN_END_DATE = pd.to_datetime('2015-01-01')

VAL_START_DATE = pd.to_datetime('2015-01-01')
VAL_END_DATE = pd.to_datetime('2015-02-01')

TEST_START_DATE = pd.to_datetime('2015-02-01')
TEST_END_DATE = pd.to_datetime('2015-07-31')



database_name = "rossmann"
metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
tables, metadata = remove_sdv_columns(tables, metadata)

# get date range of column Date from historical table
historical = tables['historical']

date_range = historical['Date'].min(), historical['Date'].max()
print(f"Date range: {date_range}")

train_mask = (historical['Date'] >= TRAIN_START_DATE) & (historical['Date'] < TRAIN_END_DATE)
val_mask = (historical['Date'] >= VAL_START_DATE) & (historical['Date'] < VAL_END_DATE)
test_mask = (historical['Date'] >= TEST_START_DATE) & (historical['Date'] <= TEST_END_DATE)

# split data and save to Path(DATA_DIR) / database_name / split / {train, val, test}.csv
split_dir = Path(DATA_DIR) / database_name / 'split'
split_dir.mkdir(parents=True, exist_ok=True)

for split in ['train', 'val', 'test']:
    (split_dir / split).mkdir(parents=True, exist_ok=True)

train = historical[train_mask].sort_values('Date')
val = historical[val_mask].sort_values('Date')
test = historical[test_mask].sort_values('Date')

train.to_csv(split_dir / 'train' / 'historical.csv', index=False)
tables['store'].to_csv(split_dir / 'train' / 'store.csv', index=False)

val.to_csv(split_dir / 'val' / 'historical.csv', index=False)
tables['store'].to_csv(split_dir / 'val' / 'store.csv', index=False)

test.to_csv(split_dir / 'test' / 'historical.csv', index=False)
tables['store'].to_csv(split_dir / 'test' / 'store.csv', index=False)

print("Data split and saved!")