import os
import pickle
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from realog.utils.metadata import Metadata
from realog.utils.data import load_tables, remove_sdv_columns, make_column_names_unique
from realog.database_to_subgraphs import create_subgraphs

DATA_DIR = "./data"


def save_subgraphs(database_name, split, subgraphs, pks, tables, metadata):
    with open(f'{DATA_DIR}/{database_name}/{split}_subgraphs.pkl', 'wb') as f:
        pickle.dump(subgraphs, f)
    for table in pks.keys():
        primary_key_column = metadata.get_primary_key(table)
        split_table = tables[table][tables[table][primary_key_column].isin(pks[table])]
        os.makedirs(f'{DATA_DIR}/{database_name}/split/{split}', exist_ok=True)
        split_table.to_csv(f'{DATA_DIR}/{database_name}/split/{split}/{table}.csv', index=False)


if __name__ == '__main__':
    

    # database_name = 'financial_v1'
    # target_table = 'loan'
    # target_column = 'loan_status'
    # train_range = (pd.to_datetime('1993-01-01'), pd.to_datetime('1998-01-01'))
    # val_range   = (pd.to_datetime('1998-01-01'), pd.to_datetime('1998-06-01'))
    # test_range  = (pd.to_datetime('1998-06-01'), pd.to_datetime('1999-01-01'))
    # task = 'classification'

    database_name = 'rossmann'
    target_table = 'historical'
    target_column = 'historical_Customers'
    task = 'regression'
    train_range = (pd.to_datetime('2014-01-01'), pd.to_datetime('2015-01-01'))
    val_range   = (pd.to_datetime('2015-01-01'), pd.to_datetime('2015-02-01'))
    test_range  = (pd.to_datetime('2015-02-01'), pd.to_datetime('2015-07-31'))


    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')

    tables = load_tables(Path(f'{DATA_DIR}/{database_name}'), metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    tables, metadata = make_column_names_unique(tables, metadata)

    if database_name == 'financial_v1':
        # map statues C -> A and D -> B
        tables['loan'].loc[tables['loan']['loan_status'] == 'C', 'loan_status'] = 'A'
        tables['loan'].loc[tables['loan']['loan_status'] == 'D', 'loan_status'] = 'B'
        # set the categories for the target column
        tables['loan']['loan_status'] = pd.Categorical(tables['loan']['loan_status'], categories=['A', 'B'])

    categories = dict()
    # TODO: for std and means we should probably only use the training data (should split the target table pks before)
    means = dict()
    stds = dict()
    for table in metadata.get_tables():
        categories[table] = {}
        means[table] = {}
        stds[table] = {}
        
        categorical_columns = metadata.get_column_names(table, sdtype='categorical')
        for column in categorical_columns:
            if tables[table][column].isna().sum() > 0:
                tables[table][column] = tables[table][column].cat.add_categories('missing')
                tables[table][column] = tables[table][column].fillna('missing')
            categories[table][column] = tables[table][column].unique().tolist()
            if task == 'classification' and table == target_table and column == target_column:
                categories[table][column].append('target')

        numerical_columns = metadata.get_column_names(table, sdtype='numerical')
        for column in numerical_columns:
            means[table][column] = tables[table][column].dropna().mean()
            stds[table][column] = tables[table][column].dropna().std()

    # split the target table into train, test and validation
    if len(metadata.get_column_names(target_table, sdtype='datetime')) > 0:
        date_column = metadata.get_column_names(target_table, sdtype='datetime')[0]
        train_mask = (tables[target_table][date_column] >= train_range[0]) & (tables[target_table][date_column] < train_range[1])
        val_mask = (tables[target_table][date_column] >= val_range[0]) & (tables[target_table][date_column] < val_range[1])
        test_mask = (tables[target_table][date_column] >= test_range[0]) & (tables[target_table][date_column] <= test_range[1])
        train_index = tables[target_table][train_mask].index
        val_index = tables[target_table][val_mask].index
        test_index = tables[target_table][test_mask].index
    else:
        train_index, test_index = train_test_split(tables[target_table].index, test_size=0.1, random_state=1)
        train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=1)
    
    train_subgraphs, train_primary_keys = create_subgraphs(train_index, tables, metadata, target_table, target_column, categories, means, stds, task=task)
    save_subgraphs(database_name, 'train', train_subgraphs, train_primary_keys, tables, metadata)
    val_subgraphs, val_primary_keys = create_subgraphs(val_index, tables, metadata, target_table, target_column, categories, means, stds, task=task)
    save_subgraphs(database_name, 'val', val_subgraphs, val_primary_keys, tables, metadata)
    test_subgraphs, test_primary_keys = create_subgraphs(test_index, tables, metadata, target_table, target_column, categories, means, stds, task=task)
    save_subgraphs(database_name, 'test', test_subgraphs, test_primary_keys, tables, metadata)

