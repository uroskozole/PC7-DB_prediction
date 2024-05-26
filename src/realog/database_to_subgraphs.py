import torch
import pandas as pd
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData


def create_graph_tables(idx, tables, metadata, target_table, use_historical=True):
    graph_tables = {}
    target_row = tables[target_table].iloc[idx]
    date_columns = metadata.get_column_names(target_table, sdtype='datetime')
    dates = target_row[date_columns]
    if len(dates) > 0:
        date = dates.values[0]
    graph_tables[target_table] = pd.DataFrame(target_row).T
    target_primary_key = metadata.get_primary_key(target_table)
    target_pk = target_row[target_primary_key]

    # TODO: this assumes only a single parent table (up to the root table)
    # This should probably be solved using recursion
    parent_tables = list(metadata.get_parents(target_table))
    already_merged_tables = set()
    current_table = target_table
    while len(parent_tables):
        parent_table = parent_tables[0]
        foreign_key = metadata.get_foreign_keys(parent_table, current_table)[0]
        parent_pk = metadata.get_primary_key(parent_table)
        fks = graph_tables[current_table][foreign_key]
        parent_table_rows = tables[parent_table][tables[parent_table][parent_pk].isin(fks)]
        graph_tables[parent_table] = parent_table_rows
        already_merged_tables.add(parent_table)
        current_table = parent_table
        parent_tables = list(metadata.get_parents(current_table))

    relationships = metadata.relationships.copy()
    while len(relationships) > 0:
        parent_table = relationships[0]['parent_table_name']
        child_table = relationships[0]['child_table_name']
        parent_pk = relationships[0]['parent_primary_key']
        child_fk = relationships[0]['child_foreign_key'] 
        # as we have already traversed from the target table to the root table,
        # we can now go down, by only mergin from the parent table to the child table
        if child_table == target_table and use_historical == False:
            pass
        elif parent_table in already_merged_tables:
            parent_pks = graph_tables[parent_table][parent_pk]
            child_table_rows = tables[child_table][tables[child_table][child_fk].isin(parent_pks)]
            date_columns = metadata.get_column_names(child_table, sdtype='datetime')
            if len(date_columns) > 0:
                child_table_rows = child_table_rows[(child_table_rows[date_columns] < date).all(axis=1)]
            if child_table in graph_tables:
                graph_tables[child_table] = pd.concat([graph_tables[child_table], child_table_rows])
            else:
                graph_tables[child_table] = child_table_rows
            already_merged_tables.add(child_table)
        else:
            relationships.append(relationships.pop(0))
            continue

        relationships.pop(0)

    primary_keys = {}
    for table in metadata.get_tables():
        table_primary_key = metadata.get_primary_key(table)
        graph_tables[table].reset_index(drop=True, inplace=True)
        primary_keys[table] = graph_tables[table][table_primary_key].tolist()

    return graph_tables, target_pk, primary_keys


def tables_to_heterodata(tables, target_table_name, target_column, target_pk, metadata, categories, means, stds, task='regression'):
    
    target_table = tables[target_table_name]
    target_primary_key = metadata.tables[target_table_name].primary_key

    if task == 'regression':
        y = target_table.loc[target_table[target_primary_key] == target_pk, target_column].values[0]
        target_table.loc[target_table[target_primary_key] == target_pk, target_column] = means[target_table_name][target_column]
    elif task == 'classification':
        y = target_table.loc[target_table[target_primary_key] == target_pk, target_column].values[0]
        y = categories[target_table_name][target_column].index(y)
        target_table.loc[target_table[target_primary_key] == target_pk, target_column] = 'target'
        # set the target column to categorical
        target_table[target_column] = pd.Categorical(target_table[target_column], categories=categories[target_table_name][target_column])

    data = HeteroData()

    # tranform the data to all numerical values
    for key in metadata.get_tables():
        if tables[key].empty:
            continue
        id_cols = metadata.get_column_names(key, sdtype='id')
        temp_table = tables[key].drop(columns=id_cols)

        categorical_columns = metadata.get_column_names(key, sdtype='categorical')
        for column in categorical_columns:
            if 'missing' in categories[key][column]:
                # add missing category
                temp_table[column] = pd.Categorical(temp_table[column], categories=categories[key][column])
        temp_table = pd.get_dummies(temp_table, columns=categorical_columns)

        # fill missing values and standardize
        for numerical_column, mean in means[key].items():
            temp_table[numerical_column] = temp_table[numerical_column].astype('float64').fillna(mean)
            temp_table[numerical_column] = (temp_table[numerical_column] - mean) / stds[key][numerical_column]

        datetime_columns = metadata.get_column_names(key, sdtype='datetime')
        for column in datetime_columns:
            nulls = temp_table[column].isnull()
            temp_table[column] = pd.to_datetime(temp_table[column], errors='coerce')
            # Do not use the year as we train on different years
            # temp_table[f'{column}_Year'] = temp_table[column].dt.year / 2000
            temp_table[f'{column}_Month'] = temp_table[column].dt.month / 12
            temp_table[f'{column}_Day'] = temp_table[column].dt.day    / 31
            # temp_table.loc[nulls, f'{column}_Year'] = 0
            temp_table.loc[nulls, f'{column}_Month'] = 0
            temp_table.loc[nulls, f'{column}_Day'] = 0
            # TODO: do hours, seconds etc.
            temp_table = temp_table.drop(columns=[column])

        # combine the transformed data to the id columns
        tables[key] = pd.concat([tables[key][id_cols], temp_table], axis=1)
        


    # Transform the ids to 0, 1, 2, ...
    id_map = {}

    for parent_table_name in metadata.get_tables():
        primary_key = metadata.get_primary_key(parent_table_name)

        if parent_table_name not in id_map:
            id_map[parent_table_name] = {}
        
        if primary_key not in id_map[parent_table_name]:
            id_map[parent_table_name][primary_key] = {}
            idx = 0
            for primary_key_val in tables[parent_table_name][primary_key].unique():
                if parent_table_name == target_table_name and primary_key_val == target_pk:
                    assert idx == 0 # the target node should always be the first node
                id_map[parent_table_name][primary_key][primary_key_val]  = idx
                idx += 1 

        for relationship in metadata.relationships:
            if relationship['parent_table_name'] != parent_table_name:
                continue
            if relationship['child_table_name'] not in id_map:
                id_map[relationship['child_table_name']] = {}
            
            id_map[relationship['child_table_name']][relationship['child_foreign_key']] = id_map[parent_table_name][relationship['parent_primary_key']]


    for table_name in id_map.keys():
        for column_name in id_map[table_name].keys():
            if column_name not in tables[table_name].columns:
                raise ValueError(f"Column {column_name} not found in table {table_name}")
            tables[table_name][column_name] = tables[table_name][column_name].map(id_map[table_name][column_name])


    # Set edges based on relationships.
    for relationship in metadata.relationships:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        foreign_key = relationship['child_foreign_key']

        tables[child_table] = tables[child_table].dropna(subset=[metadata.get_primary_key(child_table)])
        child_primary_key = metadata.get_primary_key(child_table)

        # some relationships can have missing foreign keys
        fks = tables[child_table][[foreign_key, child_primary_key]]
        fks = fks.dropna().astype('int64')
        # if fks.empty:
        #     print(f"Empty foreign keys for relationship {parent_table} -> {child_table}")
        data[parent_table, 'to', child_table].edge_index = torch.tensor(fks.values.T)
        data[child_table, 'from', parent_table].edge_index = torch.tensor(fks.loc[:, [child_primary_key, foreign_key]].values.T)

    # set connection to the target node
    data[target_table_name, 'to', 'target'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    # data['target', 'from', target_table_name].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # set the features for each node to the HeteroData object
    for key in metadata.get_tables():
        if tables[key].empty:
            data[key].x = torch.zeros((1, 6), dtype=torch.float32)
            continue
        id_cols = metadata.get_column_names(key, sdtype='id')
        tables[key] = tables[key].drop(columns=id_cols)
        table_values = tables[key].values.astype('float32')

        # if table_values empty, set torch.zeros
        if table_values.size == 0:
            data[key].x = torch.zeros((tables[key].shape[0], 1), dtype=torch.float32)
        else:
            data[key].x = torch.tensor(table_values, dtype=torch.float32)

    
    data['target'].x = torch.zeros((1, 1), dtype=torch.float32)
    if task == 'classification':
        data['target'].y = torch.tensor([y], dtype=torch.long)
        # subtract 1 because the target column has an additional 'target' category
        data['target'].num_classes = len(categories[target_table_name][target_column]) - 1
    elif task == 'regression':
        data['target'].y = torch.tensor(y.reshape(-1, 1).astype('float'), dtype=torch.float32)


    # add skip-connections from all tables (except the target_table) to the artificial target node
    for key in metadata.get_tables():
        # exclude the target and empty tables (foreign keys only)
        if key == target_table_name or (data[key].x.size(1) == 1 and data[key].x.sum().item() == 0):
            continue
        # connect only towards target
        pks = torch.arange(data[key].x.size(0))
        if data['target'].x.size(0) > 1:
            raise ValueError("Self connections not applicable for target table with multiple rows")
        target_keys = torch.zeros_like(pks)
        data[key, 'to', 'target'].edge_index = torch.stack([pks, target_keys], dim=0)

    transform = T.Compose([
        T.AddSelfLoops(),
        T.RemoveDuplicatedEdges(),
        T.RemoveIsolatedNodes(),
    ])

    return transform(data)


def create_subgraphs(index, tables, metadata, target_table, target_column, categories, means, stds, task='regression'):
    subgraphs = []
    split_primary_keys = {table: [] for table in tables.keys()}
    for idx in tqdm(index, desc='Creating subgraphs'):
        graph_tables, target_pk, primary_keys = create_graph_tables(idx, tables, metadata, target_table)
        heterodata = tables_to_heterodata(graph_tables, target_table, target_column, target_pk, metadata, 
                                        categories=categories, means=means, stds=stds, task=task)
        for table_name in graph_tables.keys():
            split_primary_keys[table_name].extend(primary_keys[table_name])
            split_primary_keys[table_name] = list(set(split_primary_keys[table_name]))
        subgraphs.append(heterodata)
    return subgraphs, split_primary_keys

