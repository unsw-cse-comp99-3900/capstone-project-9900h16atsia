import pandas as pd
import numpy as np


# data preprocessing
def load_data(data_file_path, y_file_path):
    data = pd.read_csv(data_file_path, delimiter='\t', header=0)
    data.set_index(data.columns[0], inplace=True)

    if 'Depth' in data.index:
        data.drop('Depth', inplace=True)
    if 'Unnamed: 47' in data.columns:
        data.drop('Unnamed: 47', axis=1, inplace=True)
    data.index = data.index.str.replace('"', '').str.strip()

    top_compounds = [
        "EC", "Temp", "Trichloroethene", "cis-1,2-dichloroethene",
        "Sulphide", "Chloride", "1,2-dichloroethane", "pH",
        "Er", "Vinyl chloride", "Chloroform", "1,1,2-trichloroethane",
    ]
    data = data.loc[top_compounds]
    data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    data.replace(0, pd.NA, inplace=True)
    data_filled = data.apply(lambda x: x.fillna(x.median()), axis=1)

    y_data = pd.read_csv(y_file_path)
    y_data = y_data.replace('ND', np.nan)
    y_data = y_data.dropna(subset=['EDC_delta13C'])
    y = y_data['EDC_delta13C'].astype(float)
    y_data_sorted = y_data.sort_values(by='EDC_delta13C', ascending=False)
    y_data_top_12 = y_data_sorted.tail(12)
    y = y_data_top_12['EDC_delta13C']

    return data_filled, y
