import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from realog.utils.metadata import Metadata
from realog.utils.data import load_tables, remove_sdv_columns


DATA_DIR = "data"
database_name = "Financial_v1"
target_table = "loan"
target_column = "status"

metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')
tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
tables, metadata = remove_sdv_columns(tables, metadata)

df_account = tables["account"]

# merge client to disp
df_disp = tables["disp"].merge(tables["client"], on="client_id", how="left")

# merge card to disp
df_disp = df_disp.merge(tables["card"], on="disp_id", how="left")

# calculate age and drop birth_date
df_disp["age"] = 2022 - df_disp["birth_date"].apply(lambda x: int(str(x).split("-")[0]))
df_disp.drop(columns=["birth_date"], inplace=True)

# calculate account age
df_disp["age"] = 2022 - df_disp["issued"].apply(lambda x: int(str(x).split("-")[0]))
df_disp.drop(columns=["issued"], inplace=True)

## agregate new disp and merge on account
# number of clients in each account
n_clients = df_disp.groupby("account_id").size()
df_disp = df_disp.merge(pd.DataFrame(n_clients, columns=["n_clients"]), on="account_id", how="left")

# gender counts in each account
gender_counts = df_disp.groupby(["account_id", "gender"]).size()
male_count = pd.DataFrame(gender_counts[(slice(None),"M")], columns=["male_count"])
female_count = pd.DataFrame(gender_counts[(slice(None),"M")], columns=["female_count"])
df_disp = df_disp.merge(male_count, on="account_id", how="left")
df_disp = df_disp.merge(female_count, on="account_id", how="left")

# type_x is client type, type_y is card type
dummies = pd.get_dummies(df_disp["type_y"], prefix="type_y")
df_disp = pd.concat([df_disp, dummies], axis=1)
df_disp.drop(columns=["district_id"], inplace=True)

type_counts = df_disp.groupby("account_id")[dummies.columns].sum()
type_counts = pd.DataFrame(type_counts, columns=dummies.columns)
df_disp = df_disp.merge(type_counts, on="account_id", how="left")

df_disp = df_disp[df_disp["type_x"] == "OWNER"]

df_account = df_account.merge(df_disp, on="account_id", how="left")

## agregate trans and merge to account
df_trans = tables["trans"]

ohe_cols = ["type", "operation", "k_symbol"]
dummies_cols = []
for col in ohe_cols:
    dummies = pd.get_dummies(df_trans[col], prefix=col)
    df_trans = pd.concat([df_trans, dummies], axis=1)
    df_trans.drop(columns=[col], inplace=True)
    dummies_cols.extend(dummies.columns)

counts = df_trans.groupby("account_id")[dummies_cols].sum()
df_trans_agg = pd.DataFrame(counts, columns=dummies_cols)
# df_trans = df_trans.merge(counts, on="account_id", how="left")

balance_std = df_trans.groupby("account_id")["balance"].std()
# balance_std_df = pd.DataFrame(balance_std, columns=["balance_std"])
df_trans_agg = df_trans_agg.merge(balance_std, on="account_id", how="left")

# df_trans = df_trans.merge(balance_std, on="account_id", how="left")
# df_trans.drop("balance", axis=1, inplace=True)

amount = df_trans.groupby("account_id")["amount"].sum()
# amount = pd.DataFrame(amount, columns=["amount_sum"])
df_trans_agg = df_trans_agg.merge(amount, on="account_id", how="left")

df_account = df_account.merge(df_trans_agg, on="account_id", how="left")

## aggregate order and merge to account
df_order = tables["order"]
df_order_agg = df_order.groupby("account_id")["amount"].sum()

df_account = df_account.merge(df_order_agg, on="account_id", how="left")

## merge final account to loan
df_loan = tables["loan"]
df_loan = df_loan.merge(df_account, on="account_id", how="left")

drop_cols = ["loan_id", 
             "account_id", 
             "date_x", 
             "date_y", 
             "disp_id", 
             "client_id", 
             "type_x", 
             "type_y",
             "card_id",
             ]

ohe_cols = ["gender", "frequency", "district_id"]
num_cols = ["amount_x", "amount_y", "balance", "n_clients", "age", "payments", "duration", "amount"]

df_loan.drop(columns=drop_cols, inplace=True)

# one-hot encode ohe_cols
for col in ohe_cols:
    dummies = pd.get_dummies(df_loan[col], prefix=col)
    df_loan = pd.concat([df_loan, dummies], axis=1)
    df_loan.drop(columns=col, inplace=True)

# standardize num_cols
scaler = StandardScaler()
df_loan[num_cols] = scaler.fit_transform(df_loan[num_cols])

X, y = df_loan.drop(columns=[target_column]), df_loan[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# map the target column from [A, B, C, D] to [0, 1, 2, 3]
y_train = y_train.map({"A": 0, "B": 1, "C": 2, "D": 3})
y_test = y_test.map({"A": 0, "B": 1, "C": 2, "D": 3})


model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse_test = accuracy_score(y_test, y_pred)
mse_train = accuracy_score(y_train, model.predict(X_train))

print(f"Train RMSE: {np.sqrt(mse_train)}")
print(f"Test RMSE: {np.sqrt(mse_test)}")

pass
# do preprocessing on loan

# make predictions