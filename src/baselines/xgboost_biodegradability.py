import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from realog.utils.metadata import Metadata
from realog.utils.data import load_tables, remove_sdv_columns
from realog.utils.data import denormalize_tables

DATA_DIR = "data"
database_name = "Biodegradability_v1"
target_table = "molecule"
target_column = "activity"

# data/Biodegradability_v1/metadata.json

metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')
tables_train = load_tables(f'{DATA_DIR}/{database_name}/split/train', metadata)
tables_test = load_tables(f'{DATA_DIR}/{database_name}/split/test', metadata)
tables_train, metadata_train = remove_sdv_columns(tables_train, metadata)
tables_test, metadata_test = remove_sdv_columns(tables_test, metadata)
# y = tables[target_table].pop(target_column)

denormalized_table_train = denormalize_tables(tables_train, metadata_train)
denormalized_table_test = denormalize_tables(tables_test, metadata_test)
# rename column type_bond
# denormalized_table = denormalized_table.rename(columns={9: "type_bond_1", 10: "type_bond_2"}, errors="raise")
# rename duplicately-named columns
df_train = denormalized_table_train["type_bond"]
first_train = df_train.iloc[:, 0]
second_train = df_train.iloc[:, 0]

denormalized_table_train["type_bond_1"] = first_train
denormalized_table_train["type_bond_2"] = second_train
denormalized_table_train = denormalized_table_train.drop(columns=["type_bond"])


# sample denormalized_table
# denormalized_table = denormalized_table.sample(frac=0.1, axis=0)

y_train = denormalized_table_train[target_column]
denormalized_table_train = denormalized_table_train.drop(columns=[target_column])

df_test = denormalized_table_test["type_bond"]
first_test = df_test.iloc[:, 0]
second_test = df_test.iloc[:, 0]

denormalized_table_test["type_bond_1"] = first_test
denormalized_table_test["type_bond_2"] = second_test
denormalized_table_test = denormalized_table_test.drop(columns=["type_bond"])


# sample denormalized_table
# denormalized_table = denormalized_table.sample(frac=0.1, axis=0)

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

# X_train = preprocessor.fit_transform(denormalized_table_train)
X = preprocessor.fit_transform(denorm_entire)
X_train = X[:len(denormalized_table_train)]
X_test = X[len(denormalized_table_train):]

# X_train[id_col] = idxs_train
# X_test[id_col] = idxs_test

model = XGBRegressor()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

results = pd.DataFrame(y_pred, columns=["y_pred"])
results["id_"] = idxs_test
results["truth"] = y_test
results["square_error"] = (results["y_pred"] - results["truth"]) ** 2# / results.shape[0]

# mse = mean_squared_error(y_test, y_pred)
mse = results.groupby("id_")["square_error"].mean()
mse = mse.mean()

print("RMSE on test: ", np.sqrt(mse))

# y_pred_ = model.predict(X_train)

# mse_ = mean_squared_error(y_train, y_pred_)

# print("RMSE: ", np.sqrt(mse_))

