from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import logging
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import os
from reportlab.lib.utils import ImageReader
from datetime import datetime
import time


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

def wrap_text(text, canvas, max_width):
    """Wrap text to fit within a specified width when drawing to a PDF."""
    lines = []
    words = text.split()
    current_line = ""
    for word in words:
        if canvas.stringWidth(current_line + " " + word) <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    return lines

UPLOAD_FOLDER = './uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'data_file' not in request.files:
        return jsonify({'error': 'No data file part'}), 400

    data_file = request.files['data_file']
    data_file_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    data_file.save(data_file_path)
    logging.info(f'Data file uploaded: {data_file_path}')
    return jsonify({'message': 'Data file uploaded successfully'})

@app.route('/analysis', methods=['POST'])
def predict():
    if 'data_file' not in request.files or 'y_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    data_file = request.files['data_file']
    y_file = request.files['y_file']

    data_file_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    y_file_path = os.path.join(UPLOAD_FOLDER, y_file.filename)

    data_file.save(data_file_path)
    y_file.save(y_file_path)
    logging.info(f'Data file uploaded: {data_file_path}')
    logging.info(f'Y file uploaded: {y_file_path}')

    # 读取数据文件
    data = pd.read_csv(data_file_path, delimiter='\t', header=0)
    data.set_index(data.columns[0], inplace=True)

    # 数据预处理
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

    # 读取y值文件
    y_data = pd.read_csv(y_file_path)
    y_data = y_data.replace('ND', np.nan)
    y_data = y_data.dropna(subset=['EDC_delta13C'])
    y_data_sorted = y_data.sort_values(by='EDC_delta13C', ascending=False)
    y_data_top_12 = y_data_sorted.tail(12)
    y = y_data_top_12['EDC_delta13C']

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(data_filled, y, test_size=0.3, random_state=42)
    start_time = time.time()
    # Build a GBR/GBDT model with 100 estimators, a learning rate of 0.1, and a depth of 3.
    gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbdt.fit(X_train, y_train)

    # Use GBDT model to predict and get the results.
    y_train_pred_gbdt = gbdt.predict(X_train)
    y_test_pred_gbdt = gbdt.predict(X_test)

    # Integrate the predictions of GBDT into a new feature set.
    X_train_blend = np.hstack((X_train, y_train_pred_gbdt.reshape(-1, 1)))
    X_test_blend = np.hstack((X_test, y_test_pred_gbdt.reshape(-1, 1)))

    # Build a RandomForest model.
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_blend, y_train)

    # Use the RandomForest model to predict and get the results.
    y_train_pred_rf = rf.predict(X_train_blend)
    y_test_pred_rf = rf.predict(X_test_blend)

    # Integrate the predictions of both GBDT and RandomForest into a new feature set.
    X_train_stack = np.vstack((y_train_pred_gbdt, y_train_pred_rf)).T
    X_test_stack = np.vstack((y_test_pred_gbdt, y_test_pred_rf)).T

    # Build a Ridge regression model as the second layer model with regularization.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_stack, y_train)

    # Use the Ridge model to predict and get the results.
    y_train_pred_ridge = ridge.predict(X_train_stack)
    y_test_pred_ridge = ridge.predict(X_test_stack)
    end_time = time.time()
    prediction_time = end_time - start_time
    # Evaluate performance on both training and test sets.
    metrics = {
        "Mean Squared Error": mean_squared_error,
        "Root Mean Squared Error": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "Mean Absolute Error": mean_absolute_error,
        "R² Score": r2_score,
        "Explained Variance Score": explained_variance_score,
        "Median Absolute Error": median_absolute_error,
    }

    # Collect the results
    results = []
    results.append("Training Performance:")

    for name, metric in metrics.items():
        results.append(f"{name}: {metric(y_train, y_train_pred_ridge)}")

    results.append("Testing Performance:")
    for name, metric in metrics.items():
        results.append(f"{name}: {metric(y_test, y_test_pred_ridge)}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_test_pred_ridge)
    plt.xlabel('Actual EDC_delta13C')
    plt.ylabel('Predicted EDC_delta13C')
    plt.title('Actual vs Predicted EDC_delta13C')
    plt.grid(True)
    scatter_plot_path = os.path.join(UPLOAD_FOLDER, 'scatter_plot.png')
    plt.savefig(scatter_plot_path)
    plt.close()
    explainer = shap.Explainer(ridge, X_test_stack)
    shap_values = explainer(X_test_stack)
    shap_plot_path = './shap_plot.png'
    shap.summary_plot(shap_values, X_test_stack, plot_type="bar", show=False)
    plt.savefig(shap_plot_path)
    plt.close()
    shap_summary_plot_path = './shap_summary_plot.png'
    shap.summary_plot(shap_values, X_test_stack, show=False)
    plt.savefig(shap_summary_plot_path)
    plt.close()

    # Create the PDF
    pdf_buffer = BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    y_pos = 650
    prediction_time_str = f"Prediction Time: {prediction_time:.2f} seconds"
    p.drawString(100, 750, prediction_time_str)
    for line in results:
        p.drawString(100, y_pos, line)
        y_pos -= 20
        if y_pos < 100:  # Create a new page if the space is not enough
            p.showPage()
            y_pos = 750
    p.showPage()
    p.drawImage(ImageReader(scatter_plot_path), 50, 400, width=500, height=300)
    p.showPage()
    p.drawImage(ImageReader(shap_plot_path), 50, 400, width=500, height=150)
    p.drawImage(ImageReader(shap_summary_plot_path), 50, 600, width=500, height=150)
    p.showPage()


    p.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name='prediction_results.pdf')

@app.route('/predict', methods=['POST'])
def analyse():
    if 'data_file' not in request.files or 'y_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    data_file = request.files['data_file']
    y_file = request.files['y_file']

    data_file_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    y_file_path = os.path.join(UPLOAD_FOLDER, y_file.filename)

    data_file.save(data_file_path)
    y_file.save(y_file_path)
    logging.info(f'Data file uploaded: {data_file_path}')
    logging.info(f'Y file uploaded: {y_file_path}')

    # 读取数据文件
    data = pd.read_csv(data_file_path, delimiter='\t', header=0)
    data.set_index(data.columns[0], inplace=True)

    # 数据预处理
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

    # 读取y值文件
    y_data = pd.read_csv(y_file_path)
    y_data = y_data.replace('ND', np.nan)
    y_data = y_data.dropna(subset=['EDC_delta13C'])
    y_data_sorted = y_data.sort_values(by='EDC_delta13C', ascending=False)
    y_data_top_12 = y_data_sorted.tail(12)
    y = y_data_top_12['EDC_delta13C']

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(data_filled, y, test_size=0.3, random_state=42)

    # Build a GBR/GBDT model with 100 estimators, a learning rate of 0.1, and a depth of 3.
    gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbdt.fit(X_train, y_train)

    # Use GBDT model to predict and get the results.
    y_train_pred_gbdt = gbdt.predict(X_train)
    y_test_pred_gbdt = gbdt.predict(X_test)

    # Integrate the predictions of GBDT into a new feature set.
    X_train_blend = np.hstack((X_train, y_train_pred_gbdt.reshape(-1, 1)))
    X_test_blend = np.hstack((X_test, y_test_pred_gbdt.reshape(-1, 1)))

    # Build a RandomForest model.
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_blend, y_train)

    # Use the RandomForest model to predict and get the results.
    y_train_pred_rf = rf.predict(X_train_blend)
    y_test_pred_rf = rf.predict(X_test_blend)

    # Integrate the predictions of both GBDT and RandomForest into a new feature set.
    X_train_stack = np.vstack((y_train_pred_gbdt, y_train_pred_rf)).T
    X_test_stack = np.vstack((y_test_pred_gbdt, y_test_pred_rf)).T

    # Build a Ridge regression model as the second layer model with regularization.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_stack, y_train)

    # Use the Ridge model to predict and get the results.
    y_train_pred_ridge = ridge.predict(X_train_stack)
    y_test_pred_ridge = ridge.predict(X_test_stack)



    # Create the PDF
    pdf_buffer = BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    y_pos = 750


    results = []
    y_test = y_test.astype(float)  # Ensure y_test is float type
    y_test_pred_ridge = np.array(y_test_pred_ridge).astype(float)  # Ensure predictions are float type
    errors = np.abs(y_test.values - y_test_pred_ridge)  # Use .values to get NumPy array
    high_error_indices = np.argsort(errors)
    logging.info(f'high_error_indices length: {len(data)}')
    for idx in high_error_indices:
        results.append(f"Instance {idx}:")
        results.append(f"  Actual value: {y_test.iloc[idx]}")
        results.append(f"  Predicted value: {y_test_pred_ridge[idx]}")
        results.append(f"  Error: {errors[idx]}")
        results.append(f"  Feature values: {X_test.iloc[idx].to_dict()}")

    max_width = 500  # Max width for text
    for line in results:
        wrapped_lines = wrap_text(line, p, max_width)
        for wrapped_line in wrapped_lines:
            p.drawString(50, y_pos, wrapped_line)
            y_pos -= 15
            if y_pos < 100:  # Create a new page if the space is not enough
                p.showPage()
                y_pos = 750
    p.save()
    pdf_buffer.seek(0)

    return send_file(pdf_buffer, as_attachment=True, download_name='prediction_results.pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




