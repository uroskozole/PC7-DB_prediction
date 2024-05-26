import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# from utils.metadata import Metadata
# from utils.data import load_tables, remove_sdv_columns


# def process_rossmann(tables, metadata, columns=None):
#     df = tables['historical'].merge(tables['store'], on='Store')
#     numerical_columns = [] 
#     for table in metadata.get_tables():
#         table_metadata = metadata.get_table_meta(table)
#         for column, column_info in table_metadata['columns'].items():
#             if column_info['sdtype'] == 'numerical':
#                 numerical_columns.append(column)
#             elif column_info['sdtype'] == 'id':
#                 if column in df.columns:
#                     df.drop(columns=[column], inplace=True)

#     df['StateHoliday'] = pd.Categorical(df['StateHoliday'], categories=['0', 'a'], ordered=True).codes

#     df[numerical_columns] = df[numerical_columns].fillna(-1)
#     y = df.pop('Customers')

#     if columns is not None:
#         X = df[columns]
#     else:
#         X = df
#     return X, y

# metadata = Metadata().load_from_json(f'data/rossmann_subsampled/metadata.json')

# tables_train = load_tables(f'data/rossmann_subsampled/', metadata)
# tables_train, metadata = remove_sdv_columns(tables_train, metadata)

# tables_test = load_tables(f'data/rossmann_subsampled/test_data/', metadata)
# tables_test, metadata = remove_sdv_columns(tables_test, metadata)

# df_train, y_train = process_rossmann(tables_train, metadata)
# df_test, y_test = process_rossmann(tables_test, metadata)
##############################################
data_path = "data/rossmann_subsampled/"

df_store_train = pd.read_csv(data_path+"store.csv")
df_historical_train = pd.read_csv(data_path+"historical.csv")

df_train = df_store_train.join(df_historical_train, on="Store", lsuffix='_store', rsuffix='_historical', how="right").drop(["Store_historical", "Store_store"], axis=1)

train_size = df_train.shape[0]

df_store_test = pd.read_csv(data_path+"test_data/store.csv")
df_historical_test = pd.read_csv(data_path+"test_data/historical.csv")
df_historical_train["StateHoliday"] = pd.Categorical(df_historical_train['StateHoliday'], categories=['0', 'a'], ordered=True).codes
df_historical_test["StateHoliday"] = pd.Categorical(df_historical_test['StateHoliday'], categories=['0', 'a'], ordered=True).codes

df_test = df_store_test.join(df_historical_test, on="Store", lsuffix='_store', rsuffix='_historical', how="right").drop(["Store_historical", "Store_store"], axis=1)
# df = df.fillna(df.mean())

categorical_cols = ['Promo2',
                    'CompetitionOpenSinceYear', 
                    'Assortment', 
                    'StoreType', 
                    'PromoInterval', 
                    'StateHoliday']
numerical_cols = ['CompetitionOpenSinceYear', 
                  'CompetitionOpenSinceMonth',
                   'CompetitionDistance',
                   'Promo2SinceYear',
                   'Promo2SinceWeek',
                   'DayOfWeek',
                   'Promo',
                   'Open',
                   'SchoolHoliday']# + ['Customers']   # List of numerical column names

id_cols = ['Store', 'Id']
drop_cols = ["Date"] + id_cols

df = pd.concat([df_train, df_test])
df = df.drop(drop_cols, axis=1)

if "Customers" not in numerical_cols:
    y_col = df["Customers"]
    df.drop("Customers", axis=1, inplace=True)

# Define the transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical columns
        ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical columns
    ])

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply the transformations to the DataFrame
transformed_df_ = pipeline.fit_transform(df).toarray()
transformed_df = pd.DataFrame(transformed_df_, columns=numerical_cols + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()))

model = XGBRegressor()

if "Customers" not in numerical_cols:
    X_train, y_train = transformed_df.head(train_size), y_col.head(train_size)
    X_test, y_test = transformed_df.tail(transformed_df.shape[0] - train_size), y_col.tail(transformed_df.shape[0] - train_size)
else:
    X_train, y_train = transformed_df.drop("Customers", axis=1).head(train_size), transformed_df["Customers"].head(train_size)
    X_test, y_test = transformed_df.drop("Customers", axis=1).tail(transformed_df.shape[0] - train_size), transformed_df["Customers"].tail(transformed_df.shape[0] - train_size)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("RMSE: ", np.sqrt(mse))

y_pred_ = model.predict(X_train)

mse_ = mean_squared_error(y_train, y_pred_)

print("RMSE: ", np.sqrt(mse_))
