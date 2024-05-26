import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_financial(tables, target_column="status"):
    df_account = tables["account"]
    df_account.drop(columns=["date"], inplace=True)

    # merge client to disp
    df_disp = tables["disp"].merge(tables["client"], on="client_id", how="left")

    # merge card to disp
    df_disp = df_disp.merge(tables["card"], on="disp_id", how="left")

    # calculate age and drop birth_date
    df_disp["age"] = 2022 - df_disp["birth_date"].apply(lambda x: int(str(x).split("-")[0]))
    df_disp.drop(columns=["birth_date"], inplace=True)

    # calculate account age
    df_disp["card_age"] = 2022 - df_disp["issued"].apply(lambda x: int(str(x).split("-")[0]))
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

    ## merge this account to loan
    df_loan = tables["loan"]
    # rename date column
    df_loan.rename(columns={"date": "date_loan"}, inplace=True)
    df_loan.rename(columns={"amount": "amount_loan"}, inplace=True)

    df_loan = df_loan.merge(df_account, on="account_id", how="left")

    ## agregate trans and merge to loan
    df_trans = tables["trans"]

    ohe_cols = ["type", "operation", "k_symbol"]
    dummies_cols = []
    for col in ohe_cols:
        dummies = pd.get_dummies(df_trans[col], prefix=col)
        df_trans = pd.concat([df_trans, dummies], axis=1)
        df_trans.drop(columns=[col], inplace=True)
        dummies_cols.extend(dummies.columns)

    df_loan_trans = df_loan.merge(df_trans, on="account_id", how="left")
    df_loan_trans["time_flag"] = df_loan_trans["date_loan"] > df_loan_trans["date"]
    df_loan_trans = df_loan_trans[df_loan_trans["time_flag"]]

    current_balance = df_loan_trans.groupby("account_id").apply(lambda x: x.sort_values(by='date').iloc[0])["balance"]

    # df_loan_agg = df_loan.groupby("account_id").sum()
    counts = df_loan_trans.groupby("account_id")[dummies_cols].sum()
    counts = pd.DataFrame(counts, columns=dummies_cols)

    #balance_std = df_loan_trans.groupby("account_id")["balance"].std()

    amount = df_loan_trans.groupby("account_id")["amount"].sum()
    # amount = pd.DataFrame(amount, columns=dummies_cols)

    df_loan = df_loan.merge(counts, on="account_id", how="left")

    # df_loan.drop(columns=["balance"], inplace=True)
    # df_loan = df_loan.merge(balance_std, on="account_id", how="left")
    df_loan = df_loan.merge(current_balance, on="account_id", how="left")

    # df_loan.drop(columns=["amount"], inplace=True)
    df_loan = df_loan.merge(amount, on="account_id", how="left")

    ## aggregate order and merge to loan
    df_order = tables["order"]
    df_order_agg = df_order.groupby("account_id")["amount"].sum()

    df_loan = df_loan.merge(df_order_agg, on="account_id", how="left")

    ## process the dataset
    drop_cols = ["loan_id", 
                "account_id", 
                "date_loan",
                "disp_id",
                "client_id",
                "type_x",
                "card_id",
                "type_y",
                ]

    ohe_cols = ["gender", "frequency", "district_id"]
    num_cols = ["amount_x", "amount_y", "balance", "n_clients", "age", "payments", "duration", "amount_loan", "card_age"]

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
    y = y.map({"A": 0, "B": 1, "C": 0, "D": 1})
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # map the target column from [A, B, C, D] to [0, 1, 2, 3]
    # y_test = y_test.map({"A": 0, "B": 1, "C": 2, "D": 3})

    return X, y

def process_financial_split(tables, target_column="loan_status"):
    df_account = tables["account"]
    df_account.drop(columns=["account_date_account_created"], inplace=True)

    # merge client to disp
    df_disp = tables["disp"].merge(tables["client"], left_on="disp_client_id", right_on="client_client_id", how="left")
    df_disp.drop(columns=["client_client_id"], inplace=True)

    # merge card to disp
    df_disp = df_disp.merge(tables["card"], left_on="disp_disp_id", right_on="card_disp_id", how="left")
    df_disp.drop(columns=["card_disp_id"], inplace=True)

    # calculate age and drop birth_date
    df_disp["age"] = 2022 - df_disp["client_birth_date"].apply(lambda x: int(str(x).split("-")[0]) if pd.notnull(x) else pd.NaT)
    df_disp["age"].fillna(df_disp["age"].mean(), inplace=True)
    df_disp.drop(columns=["client_birth_date"], inplace=True)

    # calculate account age
    df_disp["card_age"] = 2022 - df_disp["card_issued"].apply(lambda x: int(str(x).split("-")[0]) if pd.notnull(x) else pd.NaT)
    df_disp["card_age"].fillna(df_disp["card_age"].mean(), inplace=True)
    df_disp.drop(columns=["card_issued"], inplace=True)

    ## agregate new disp and merge on account
    # number of clients in each account
    n_clients = df_disp.groupby("disp_account_id").size()
    df_disp = df_disp.merge(pd.DataFrame(n_clients, columns=["n_clients"]), on="disp_account_id", how="left")

    # gender counts in each account
    gender_counts = df_disp.groupby(["disp_account_id", "client_gender"]).size()
    male_count = pd.DataFrame(gender_counts[(slice(None),"M")], columns=["male_count"])
    female_count = pd.DataFrame(gender_counts[(slice(None),"M")], columns=["female_count"])
    df_disp = df_disp.merge(male_count, on="disp_account_id", how="left")
    df_disp = df_disp.merge(female_count, on="disp_account_id", how="left")

    # type_x is client type, type_y is card type
    dummies = pd.get_dummies(df_disp["card_type"], prefix="card_type")
    df_disp = pd.concat([df_disp, dummies], axis=1)
    df_disp.drop(columns=["client_district_id"], inplace=True)

    type_counts = df_disp.groupby("disp_account_id")[dummies.columns].sum()
    type_counts = pd.DataFrame(type_counts, columns=dummies.columns)
    df_disp = df_disp.merge(type_counts, on="disp_account_id", how="left")

    df_disp = df_disp[df_disp["disp_type"] == "OWNER"]

    df_account = df_account.merge(df_disp, left_on="account_account_id", right_on="disp_account_id", how="left")
    df_account.drop(columns=["disp_account_id"], inplace=True)

    ## merge this account to loan
    df_loan = tables["loan"]
    # rename date column
    df_loan.rename(columns={"loan_date": "date_loan"}, inplace=True)
    df_loan.rename(columns={"loan_amount": "amount_loan"}, inplace=True)

    df_loan = df_loan.merge(df_account, left_on="loan_account_id", right_on="account_account_id", how="left")
    df_loan.drop(columns=["account_account_id"], inplace=True)

    ## agregate trans and merge to loan
    df_trans = tables["trans"]

    ohe_cols = ["trans_type", "trans_operation", "trans_k_symbol"]
    dummies_cols = []
    for col in ohe_cols:
        dummies = pd.get_dummies(df_trans[col], prefix=col)
        df_trans = pd.concat([df_trans, dummies], axis=1)
        df_trans.drop(columns=[col], inplace=True)
        dummies_cols.extend(dummies.columns)

    df_loan_trans = df_loan.merge(df_trans, left_on="loan_account_id", right_on="trans_account_id", how="left")
    df_loan_trans.drop(columns=["trans_account_id"], inplace=True)

    df_loan_trans["time_flag"] = df_loan_trans["date_loan"] > df_loan_trans["trans_date_of_transaction"]
    df_loan_trans = df_loan_trans[df_loan_trans["time_flag"]]

    current_balance = df_loan_trans.groupby("loan_account_id").apply(lambda x: x.sort_values(by='trans_date_of_transaction').iloc[0])["trans_balance"]

    # df_loan_agg = df_loan.groupby("account_id").sum()
    counts = df_loan_trans.groupby("loan_account_id")[dummies_cols].sum()
    counts = pd.DataFrame(counts, columns=dummies_cols)

    #balance_std = df_loan_trans.groupby("account_id")["balance"].std()

    amount = df_loan_trans.groupby("loan_account_id")["trans_amount"].sum()
    # amount = pd.DataFrame(amount, columns=dummies_cols)

    df_loan = df_loan.merge(counts, on="loan_account_id", how="left")

    # df_loan.drop(columns=["balance"], inplace=True)
    # df_loan = df_loan.merge(balance_std, on="account_id", how="left")
    df_loan = df_loan.merge(current_balance, on="loan_account_id", how="left")

    # df_loan.drop(columns=["amount"], inplace=True)
    df_loan = df_loan.merge(amount, on="loan_account_id", how="left")

    ## aggregate order and merge to loan
    df_order = tables["order"]
    df_order_agg = df_order.groupby("order_account_id")["order_amount"].sum()

    df_loan = df_loan.merge(df_order_agg, left_on="loan_account_id", right_on="order_account_id", how="left")

    ## process the dataset
    drop_cols = ["loan_loan_id", 
                "loan_account_id", 
                "date_loan",
                "disp_disp_id",
                "disp_client_id",
                "disp_type",
                "card_card_id",
                "card_type",
                ]

    ohe_cols = ["client_gender", "account_frequency", "account_district_id"]
    num_cols = ["amount_loan", "order_amount", "trans_amount", "trans_balance", "n_clients", "age", "loan_payments", "loan_duration"]

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
    y = y.map({"A": 0, "B": 1, "C": 0, "D": 1})
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # map the target column from [A, B, C, D] to [0, 1, 2, 3]
    # y_test = y_test.map({"A": 0, "B": 1, "C": 2, "D": 3})

    return X, y
