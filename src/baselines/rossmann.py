import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--linear", action="store_true", help="Use linear regression model instead of XGBoost")
parser.add_argument("--bootstrap", action="store_true", help="Bootstrap test results")
args = parser.parse_args()

SUBSAMPLED = False

if SUBSAMPLED:
    data_path = "data/rossmann_subsampled/"

    df_store_train = pd.read_csv(data_path+"store.csv")
    df_historical_train = pd.read_csv(data_path+"historical.csv")
else:
    data_path = "data/rossmann/split/"
    df_store_train = pd.read_csv(data_path+"train/store.csv")
    df_historical_train = pd.read_csv(data_path+"train/historical.csv")

df_train = df_store_train.merge(df_historical_train, on="Store", how="right").drop(["Store"], axis=1)

train_size = df_train.shape[0]
if SUBSAMPLED:
    df_store_test = pd.read_csv(data_path+"test_data/store.csv")
    df_historical_test = pd.read_csv(data_path+"test_data/historical.csv")
else:
    df_store_test = pd.read_csv(data_path+"test/store.csv")
    df_historical_test = pd.read_csv(data_path+"test/historical.csv")

df_historical_train["StateHoliday"] = pd.Categorical(df_historical_train['StateHoliday'], categories=['0', 'a'], ordered=True).codes
df_historical_test["StateHoliday"] = pd.Categorical(df_historical_test['StateHoliday'], categories=['0', 'a'], ordered=True).codes

df_test = df_store_test.merge(df_historical_test, on="Store", how="right").drop(["Store"], axis=1)

categorical_cols = ['Promo2',
                    'CompetitionOpenSinceYear', 
                    'Assortment', 
                    'StoreType', 
                    'PromoInterval',] 
                    #'StateHoliday']
numerical_cols = ['CompetitionOpenSinceYear', 
                  'CompetitionOpenSinceMonth',
                   'CompetitionDistance',
                   'Promo2SinceYear',
                   'Promo2SinceWeek',
                   'DayOfWeek',
                   'Promo',
                   'Open',
                   'SchoolHoliday']

id_cols = ['Id']
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
transformed_df_ = pipeline.fit_transform(df)
transformed_df = pd.DataFrame(transformed_df_, columns=numerical_cols + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()))

if args.linear:
    model = LinearRegression()
    transformed_df = transformed_df.fillna(transformed_df.mean())
else:
    model = XGBRegressor()

if "Customers" not in numerical_cols:
    X_train, y_train = transformed_df.head(train_size), y_col.head(train_size)
    X_test, y_test = transformed_df.tail(transformed_df.shape[0] - train_size), y_col.tail(transformed_df.shape[0] - train_size)
else:
    X_train, y_train = transformed_df.drop("Customers", axis=1).head(train_size), transformed_df["Customers"].head(train_size)
    X_test, y_test = transformed_df.drop("Customers", axis=1).tail(transformed_df.shape[0] - train_size), transformed_df["Customers"].tail(transformed_df.shape[0] - train_size)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

if args.bootstrap:
    print("Bootstrapped results------------")
    rmses = []
    for i in range(100):
        indices = np.random.choice(range(len(y_test)), len(y_test), replace=True)
        se = [(y_t - y_p)**2 for y_t, y_p in zip(y_test[indices], y_pred[indices])]
        mse = np.mean(se)

        rmses.append(np.sqrt(mse))
    rmse = np.mean(rmses)
    std = np.std(rmses) / len(rmses)

    print(f"RMSE: {rmse} +- {std}")
else:
    print("Non-bootstrapped results------------")
    se = [(y_t - y_p)**2 for y_t, y_p in zip(y_test, y_pred)]
    mse = np.mean(se)
    std = np.std(se) / np.sqrt(len(se))

    print(f"RMSE: {np.sqrt(mse)} +- {np.sqrt(std)}")

    y_pred_ = model.predict(X_train)

    mse_ = mean_squared_error(y_train, y_pred_)

    print("RMSE: ", np.sqrt(mse_))
