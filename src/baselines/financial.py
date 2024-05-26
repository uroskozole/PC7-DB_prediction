import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import argparse	
from imblearn.over_sampling import SMOTE

from realog.utils.metadata import Metadata
from realog.utils.data import load_tables, remove_sdv_columns
from utils_baselines import process_financial, process_financial_split


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model to use for training")
parser.add_argument("--entire_data", action="store_true", help="Whether to use entire data or not")
parser.add_argument("--seed", default="42", help="Radnom seed for splitting the data")
args = parser.parse_args()

database_name = "Financial_v1"
target_table = "loan"
target_column = "status"
DATA_DIR = "data"
smote = True
validate = True

if args.model == "xgb":
    model = XGBClassifier()
elif args.model == "logistic":
    model = LogisticRegression()
else:
    raise ValueError("Model not supported") 

if args.entire_data:
    metadata = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/metadata.json')
    tables = load_tables(f'{DATA_DIR}/{database_name}/', metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    X, y = process_financial(tables)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(args.seed))

else:
    metadata_train = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/split/train/metadata_renamed.json')
    metadata_test = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/split/test/metadata_renamed.json')
    metadata_val = Metadata().load_from_json(f'{DATA_DIR}/{database_name}/split/val/metadata_renamed.json')
    tables_train = load_tables(f'{DATA_DIR}/{database_name}/split/train', metadata_train)
    tables_test = load_tables(f'{DATA_DIR}/{database_name}/split/test', metadata_train)
    tables_val = load_tables(f'{DATA_DIR}/{database_name}/split/val', metadata_train)

    X_train, y_train = process_financial_split(tables_train)
    X_test, y_test = process_financial_split(tables_test)
    X_val, y_val = process_financial_split(tables_val)
    if smote:
        smo = SMOTE(k_neighbors=6)
        X_train, y_train = smo.fit_resample(X_train, y_train)

    cols_train = X_train.columns
    cols_test = X_test.columns
    cols_val = X_val.columns

    for col in  cols_train:
        if col not in cols_test:
            X_test[col] = 0

    for col in  cols_test:
        if col not in cols_train:
            X_train[col] = 0
    
    for col in X_train.columns:
        if col not in X_val.columns:
            X_val[col] = 0

    X_test = X_test[X_train.columns]
    X_val = X_val[X_train.columns]

if args.model == "logistic":
    # fill missing values with mean
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

weight = {0 : np.sum(np.array(y_train)==1)/len(y_train), 1 : np.sum(np.array(y_train)==0)/len(y_train)}

classes_weights = class_weight.compute_sample_weight(
    class_weight=weight,
    y=y_train
)
model.fit(X_train, y_train, 
        #   sample_weight=classes_weights,
          )

if validate:
    y_pred = model.predict(X_val)

    mse_val = accuracy_score(y_val, y_pred)
    mse_train = accuracy_score(y_train, model.predict(X_train))

    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_val, y_pred, average="binary")

    # print(f"Recall: {recall}")
    # print(f"Precision: {precision}")
    print(f"F1: {f1}")

    print(f"Train accuracy: {np.sqrt(mse_train)}")
    print(f"Test accuracy: {np.sqrt(mse_val)}")


else:
    y_pred = model.predict(X_test)

    mse_test = accuracy_score(y_test, y_pred)
    mse_train = accuracy_score(y_train, model.predict(X_train))

    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")

    # print(f"Recall: {recall}")
    # print(f"Precision: {precision}")
    print(f"F1: {f1}")

    print(f"Train accuracy: {np.sqrt(mse_train)}")
    print(f"Test accuracy: {np.sqrt(mse_test)}")

