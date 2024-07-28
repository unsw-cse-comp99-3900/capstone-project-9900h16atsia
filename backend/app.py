from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
ridge = Ridge(alpha=1.0)



def train_models(X, y):

    gbdt.fit(X, y)
    y_pred_gbdt = gbdt.predict(X)


    X_blend = np.hstack((X, y_pred_gbdt.reshape(-1, 1)))


    rf.fit(X_blend, y)
    y_pred_rf = rf.predict(X_blend)


    X_stack = np.vstack((y_pred_gbdt, y_pred_rf)).T


    ridge.fit(X_stack, y)



@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Received a prediction request')
    if 'data_file' not in request.files or 'y_file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'})

    data_file = request.files['data_file']
    y_file = request.files['y_file']


    data = pd.read_csv(data_file, delimiter='\t', header=0)
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


    y_data = pd.read_csv(y_file)
    y_data = y_data.replace('ND', np.nan)
    y_data = y_data.dropna(subset=['EDC_delta13C'])
    y_data_sorted = y_data.sort_values(by='EDC_delta13C', ascending=False)
    y_data_top_12 = y_data_sorted.tail(12)
    y = y_data_top_12['EDC_delta13C']


    X_train, X_test, y_train, y_test = train_test_split(data_filled, y, test_size=0.3, random_state=42)


    train_models(X_train, y_train)


    y_pred_gbdt = gbdt.predict(X_test)
    X_blend = np.hstack((X_test, y_pred_gbdt.reshape(-1, 1)))
    y_pred_rf = rf.predict(X_blend)
    X_stack = np.vstack((y_pred_gbdt, y_pred_rf)).T
    y_pred_ridge = ridge.predict(X_stack)


    logging.info("Prediction Results: %s", y_pred_ridge)

    return jsonify({'prediction': y_pred_ridge.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




