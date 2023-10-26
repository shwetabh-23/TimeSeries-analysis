import numpy as np
import pandas as pd

def set_matrix_form_multivariate(df, window):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window):
        row = [a for a in df_as_np[i : i + window]]
        X.append(row)
        label = df_as_np[i+window][0]
        y.append(label)
    return np.array(X), np.array(y)

def set_matrix_form_univariate(df, window):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window):
        row = [[a] for a in df_as_np[i : i + window]]
        X.append(row)
        label = df_as_np[i+window]
        y.append(label)
    return np.array(X), np.array(y)

def create_multivariate_data(df):
    new_df = pd.DataFrame({'Temp' : df['T (degC)']})
    new_df['Seconds'] = new_df.index.map(pd.Timestamp.timestamp)
    day = 60*60*24
    year = 365.2425*day

    new_df['Day sin'] = np.sin(new_df['Seconds'] * (2* np.pi / day))
    new_df['Day cos'] = np.cos(new_df['Seconds'] * (2 * np.pi / day))
    new_df['Year sin'] = np.sin(new_df['Seconds'] * (2 * np.pi / year))
    new_df['Year cos'] = np.cos(new_df['Seconds'] * (2 * np.pi / year))

    new_df.drop('Seconds', axis = 1, inplace = True)
    #print(new_df.head())
    return new_df
