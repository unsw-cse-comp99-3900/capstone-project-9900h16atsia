import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor

file_path = 'EDCPCE_filled1.xlsx'
data = pd.read_excel(file_path)
print(data)

row_dropped = data.drop(data[(data['PCE_delta13C'] == 'ND') | (data['Depth'] == '#OTU ID')].index)
columns_dropped = row_dropped.drop(columns=['EDC_delta13C'])

columns_dropped = columns_dropped.drop(columns=['SampleID'])
columns_to_drop = columns_dropped.columns[(columns_dropped == 0).all()]
data_cleaned = columns_dropped.drop(columns=columns_to_drop)


data_cleaned = data_cleaned.apply(lambda x: x.fillna(x.median()), axis=0)
minmax_scaler=preprocessing.MinMaxScaler()
data=minmax_scaler.fit_transform(data_cleaned)
variances = data_cleaned.var().sort_values(ascending=False)
top_50_features = variances.head(100).index
print(top_50_features)
if 'PCE_delta13C' not in top_50_features:
    top_50_features = top_50_features.insert(0, 'PCE_delta13C')
data_top_50 = data_cleaned[top_50_features]
assert 'PCE_delta13C' in data_top_50
output_file_path = 'data_top_50.csv'
data_top_50.to_csv(output_file_path, index=False)
X = data_cleaned.iloc[:, :-1]
Y = data_cleaned['PCE_delta13C']
sfs = SFS(RandomForestRegressor(n_estimators=20, max_depth=8), k_features=25, forward=True, floating=False, scoring='r2',
          cv=0)
X = data_top_50.iloc[:, :-1]
Y = data_top_50['PCE_delta13C']
sfs.fit(X, Y)
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.grid()
plt.show()
top_19_features = list(sfs.k_feature_names_)[:20]
print(top_19_features)


data_top_19 = data_cleaned[top_19_features]


output_file_path_19 = 'data_top_20.csv'
data_top_19.to_csv(output_file_path_19, index=False)