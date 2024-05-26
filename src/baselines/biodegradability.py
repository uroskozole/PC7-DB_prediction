import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LinearRegression
import argparse

from realog.utils.metadata import Metadata
from realog.utils.data import load_tables, remove_sdv_columns
from realog.utils.data import denormalize_tables

DATA_DIR = "data"
database_name = "Biodegradability_v1"
target_table = "molecule"
target_column = "activity"

argparser = argparse.ArgumentParser()
argparser.add_argument("--bootstrap", action="store_true", help="Use the bootstrap dataset")
argparser.add_argument("--model", required=True, help="Model to use for obtaining baseline")
args = argparser.parse_args()

if args.model == "xgb":
    model = XGBRegressor()
elif args.model == "linear":
    model = LinearRegression()
else:
    raise ValueError("Model not supported! Available models: 'xgb', 'linear'")

metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')
tables_train = load_tables(f'{DATA_DIR}/{database_name}/split/train', metadata)
tables_test = load_tables(f'{DATA_DIR}/{database_name}/split/test', metadata)
tables_train, metadata_train = remove_sdv_columns(tables_train, metadata)
tables_test, metadata_test = remove_sdv_columns(tables_test, metadata)

denormalized_table_train = denormalize_tables(tables_train, metadata_train)
denormalized_table_test = denormalize_tables(tables_test, metadata_test)

df_train = denormalized_table_train["type_bond"]
first_train = df_train.iloc[:, 0]
second_train = df_train.iloc[:, 0]

denormalized_table_train["type_bond_1"] = first_train
denormalized_table_train["type_bond_2"] = second_train
denormalized_table_train = denormalized_table_train.drop(columns=["type_bond"])

y_train = denormalized_table_train[target_column]
denormalized_table_train = denormalized_table_train.drop(columns=[target_column])

df_test = denormalized_table_test["type_bond"]
first_test = df_test.iloc[:, 0]
second_test = df_test.iloc[:, 0]

denormalized_table_test["type_bond_1"] = first_test
denormalized_table_test["type_bond_2"] = second_test
denormalized_table_test = denormalized_table_test.drop(columns=["type_bond"])

y_test = denormalized_table_test[target_column]
denormalized_table_test = denormalized_table_test.drop(columns=[target_column])

ohe_cols = ["group_id", "type", "type_gmember", "type_bond_1", "type_bond_2"]
std_cols = ["mweight", "logp"]
drop_cols = ["atom_id", "atom_id", "atom_id_atom_id2", "atom_id2", "molecule_id"]
id_col = "molecule_id"

# one-hot encode and standardize
idxs_train = denormalized_table_train[id_col]
idxs_test = denormalized_table_test[id_col]


preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(), ohe_cols),
        ("std", StandardScaler(), std_cols),
        ("drop", "drop", drop_cols)
    ]
)
denorm_entire = pd.concat([denormalized_table_train, denormalized_table_test])

X = preprocessor.fit_transform(denorm_entire)
idxs = pd.concat([idxs_train, idxs_test]).values.reshape(-1, 1)



if args.bootstrap:
    y = pd.concat([y_train, y_test])
    rmses = np.array([])

    for i in range(10):

        split_idxs = np.random.permutation(X.shape[0], )
        # get a permutation of X elements
        X_train = X[split_idxs[:len(denormalized_table_train)]]
        X_test = X[split_idxs[len(denormalized_table_train):]]

        y_train = y.iloc[split_idxs[:len(denormalized_table_train)]]
        y_test = y.iloc[split_idxs[len(denormalized_table_train):]]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results = pd.DataFrame(y_pred, columns=["y_pred"])
        results["id_"] = idxs[split_idxs[len(denormalized_table_train):], 0]
        results["truth"] = y_test.values
        results["square_error"] = (results["y_pred"] - results["truth"]) ** 2

        mse = results.groupby("id_")["square_error"].mean()
        rmses = np.append(rmses, np.sqrt(mse.mean()))

        print(f"interim rmse: {np.sqrt(mse.mean())}")
    
    std = np.std(rmses)
    print(f"RMSE on test: {np.mean(rmses)} +- {std/np.sqrt(rmses.shape[0])}")
else:
    X_train = X[:len(denormalized_table_train)]
    X_test = X[len(denormalized_table_train):]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = pd.DataFrame(y_pred, columns=["y_pred"])
    results["id_"] = idxs_test
    results["truth"] = y_test
    results["square_error"] = (results["y_pred"] - results["truth"]) ** 2

    mse = results.groupby("id_")["square_error"].mean()
    print(f"RMSE on test: {np.sqrt(mse.mean())} +- {np.sqrt(np.std(mse.values)/np.sqrt(mse.shape[0]))}")