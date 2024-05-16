from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from realog.utils.metadata import Metadata
from realog.utils.data import denormalize_tables, load_tables, save_tables


def split_on_time(tables, metadata, train_range, val_range, test_range, date_columns=None):
    tables_to_merge = deepcopy(tables)
    for table in tables.keys():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata['columns'].items():
            if column_info['sdtype'] == 'id':
                # set id to int32 if possible
                if tables_to_merge[table][column].max() <= np.iinfo('int32').max:
                    tables_to_merge[table][column] = tables_to_merge[table][column].astype('int32')
                else:
                    raise ValueError(f"ID column {column} in table {table} has values greater than int32")
            elif column_info['sdtype'] == 'datetime':
                if date_columns is not None and column not in date_columns:
                    tables_to_merge[table].drop(column, axis=1, inplace=True)
                else:
                    print(f'{column.upper()} min: {tables_to_merge[table][column].min()}, max: {tables_to_merge[table][column].max()}')
            else:
                tables_to_merge[table].drop(column, axis=1, inplace=True)

    denormalized = denormalize_tables(tables_to_merge, metadata)
    if date_columns is None:
        date_columns = []
        for table in tables.keys():
            date_columns.extend(metadata.get_column_names(table, sdtype='datetime'))

    train_indices = ((denormalized[date_columns] >= train_range[0]) & (denormalized[date_columns] < train_range[1])).all(axis=1)
    val_indices = ((denormalized[date_columns] >= val_range[0]) & (denormalized[date_columns] < val_range[1])).all(axis=1)
    test_indices = ((denormalized[date_columns] >= test_range[0]) & (denormalized[date_columns] < test_range[1])).all(axis=1)
    denormalized_train = denormalized.loc[train_indices.values]
    denormalized_val = denormalized.loc[val_indices.values]
    denormalized_test = denormalized.loc[test_indices.values]

    train_tables = {}
    val_tables = {}
    test_tables = {}
    for table in tables.keys():
        primary_key = metadata.get_primary_key(table)
        train_pks = denormalized_train[primary_key].unique()
        val_pks = denormalized_val[primary_key].unique()
        test_pks = denormalized_test[primary_key].unique()
        train_tables[table] = tables[table].loc[tables[table][primary_key].isin(train_pks)]
        val_tables[table] = tables[table].loc[tables[table][primary_key].isin(val_pks)]
        test_tables[table] = tables[table].loc[tables[table][primary_key].isin(test_pks)]

    return train_tables, val_tables, test_tables


def split_train_val_test(tables, metadata, target_table, val_ratio=0.1, test_ratio=0.1, random_state=42):
    tables_to_merge = deepcopy(tables)

    for table in tables.keys():
        table_metadata = metadata.get_table_meta(table)
        for column, column_info in table_metadata['columns'].items():
            if column_info['sdtype'] == 'id':
                # set id to int32 if possible
                if tables_to_merge[table][column].dtype == 'int64' and tables_to_merge[table][column].max() <= np.iinfo('int32').max:
                    tables_to_merge[table][column] = tables_to_merge[table][column].astype('int32')
            else:
                tables_to_merge[table].drop(column, axis=1, inplace=True)
    
    target_pk = metadata.get_primary_key(target_table)
    indices = tables_to_merge[target_table][target_pk].values
    train_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=random_state)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_ratio, random_state=random_state)

    denormalized = denormalize_tables(tables_to_merge, metadata)

    
    train_idx = denormalized[target_pk].isin(train_indices)
    val_idx = denormalized[target_pk].isin(val_indices)
    test_idx = denormalized[target_pk].isin(test_indices)

    denormalized_train = denormalized.loc[train_idx]
    denormalized_val = denormalized.loc[val_idx]
    denormalized_test = denormalized.loc[test_idx]

    train_tables = {}
    val_tables = {}
    test_tables = {}
    for table in tables.keys():
        primary_key = metadata.get_primary_key(table)
        train_pks = denormalized_train[primary_key].unique()
        val_pks = denormalized_val[primary_key].unique()
        test_pks = denormalized_test[primary_key].unique()
        train_tables[table] = tables[table].loc[tables[table][primary_key].isin(train_pks)]
        val_tables[table] = tables[table].loc[tables[table][primary_key].isin(val_pks)]
        test_tables[table] = tables[table].loc[tables[table][primary_key].isin(test_pks)]

    return train_tables, val_tables, test_tables


def split_data(database_name, data_dir='data', target_table=None, train_range=None, val_range=None, test_range=None, date_columns=None, val_ratio=0.1, test_ratio=0.1, random_state=42):
    print(f"Splitting data for {database_name}")
    metadata = Metadata().load_from_json(f'{data_dir}/{database_name}/metadata.json')
    
    tables = load_tables(f'{data_dir}/{database_name}/', metadata)
    # for some reason our other files assume the split data has sdv columns removed
    # tables, metadata = remove_sdv_columns(tables, metadata)

    if target_table is not None:
        train_tables, val_tables, test_tables = split_train_val_test(tables, metadata, target_table, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state)
    elif train_range is not None:
        train_tables, val_tables, test_tables = split_on_time(tables, metadata, train_range, val_range, test_range, date_columns=date_columns)
    else:
        raise ValueError('Either target_table or train_range must be provided')
        
    
    split_dir = Path(data_dir) / database_name / 'split'
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (split_dir / split).mkdir(parents=True, exist_ok=True)
        
    save_tables(train_tables, split_dir / 'train')
    save_tables(val_tables, split_dir / 'val')
    save_tables(test_tables, split_dir / 'test')
    print("Data split and saved!")
