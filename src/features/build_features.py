import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def drop_cols(df, cols):
    df = df.drop(columns=cols)
    return df

def fill_cols(df, cols, coltype):
    if coltype.eq('num'):
        df[cols] = df[cols].fillna(df[cols].mean(), axis=0)
    elif coltype.eq('cat'):
        df[cols] = df[cols].fillna('MSNG', axis=0)
    return df[cols]

def get_dataframe(in_filepath):
    # Create DataFrame from input file
    df = pd.read_csv(in_filepath, index_col='Id')
    df = df.drop(columns=['SalePrice'], errors='ignore')
    return df

def clean_data(in_filepath):
    """
    Perform feature trimming, re-encoding, and engineering for housing data

    INPUT: Housing DataFrame
    OUTPUT: Trimmed and cleaned housing DataFrame
    """

    # Create DataFrame from input file
    df = get_dataframe(in_filepath)
    categorical_cols = df.select_dtypes(include=object).columns

    # Perform One-Hot Encoding on our Categorical Data

    df_enc = df.copy()
    onehot_df = df_enc.drop(columns=categorical_cols)
    features_onehot_enc = pd.get_dummies(df_enc[categorical_cols], dummy_na=True)
    onehot_df[features_onehot_enc.columns] = features_onehot_enc

    return onehot_df

def impute_xform(df):

    imp = IterativeImputer(missing_values=np.nan, random_state=5, max_iter=20, add_indicator=True)
    imputed_arr = imp.fit_transform(df)
    nans = df.isna().sum()
    nan_labels = nans[nans > 0].index
    nan_labels = [col + '_nan' for col in nan_labels]

    encoded = list(df.columns)
    encoded.extend(nan_labels)
    features_imputed = pd.DataFrame(imputed_arr, columns=encoded)

    skewed = ['ScreenPorch', 'PoolArea', 'LotFrontage', '3SsnPorch', 'LowQualFinSF']
    features_log_xformed = pd.DataFrame(data = features_imputed)
    features_log_xformed[skewed] = features_imputed[skewed].apply(lambda x: np.log(x + 1))

    return features_log_xformed

def scale_features(df):
    scaler = StandardScaler()
    numerical = df.select_dtypes(include=np.number).columns

    features_scaled = pd.DataFrame(data = df)
    features_scaled[numerical] = scaler.fit_transform(df[numerical])

    features_final = features_scaled.copy()

    return features_final