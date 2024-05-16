import pandas as pd

from realog.split_data import split_data

# Rossmann dataset
# train_range = (pd.to_datetime('2013-01-01'), pd.to_datetime('2015-01-01'))
# val_range = (pd.to_datetime('2015-01-01'), pd.to_datetime('2015-02-01'))
# test_range = (pd.to_datetime('2015-02-01'), pd.to_datetime('2015-07-31'))
# date_columns = ['Date']
# split_data("rossmann", train_range=train_range, val_range=val_range, test_range=test_range, date_columns=date_columns)

# # Financial dataset
train_range = (pd.to_datetime('1993-01-01'), pd.to_datetime('1997-01-01'))
val_range = (pd.to_datetime('1997-01-01'), pd.to_datetime('1998-01-01'))
test_range = (pd.to_datetime('1998-01-01'), pd.to_datetime('1999-01-01'))

# train_range = (pd.to_datetime('1993-01-01'), pd.to_datetime('1994-01-01'))
# val_range = (pd.to_datetime('1994-01-01'), pd.to_datetime('1995-01-01'))
# test_range = (pd.to_datetime('1995-01-01'), pd.to_datetime('1996-01-01'))

# date_columns = ['date']
# split_data("financial_v1", train_range=train_range, val_range=val_range, test_range=test_range, date_columns=date_columns)

# Biodegradability dataset
split_data("Biodegradability_v1", target_table='molecule', random_state=1)