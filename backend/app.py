from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import logging
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from reportlab.lib.utils import ImageReader
import time
from utils.data_processing import load_data
from utils.visualization import plot_results,plot_shap_values
from models.model import Model


app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


# Print pdf with automatic line breaks
def wrap_text(text, canvas, max_width):
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


# Upload datafile and label file
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'data_file' not in request.files:
        return jsonify({'error': 'No data file part'}), 400

    data_file = request.files['data_file']
    data_file_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
    data_file.save(data_file_path)
    logging.info(f'Data file uploaded: {data_file_path}')
    return jsonify({'message': 'Data file uploaded successfully'})


# train model and download analysis pdf file
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
    start_time = time.time()
    data_filled, y = load_data(data_file_path, y_file_path)
    model = Model()
    model.train(data_filled, y)
    end_time = time.time()
    prediction_time = end_time - start_time
    results = model.evaluate()
    plot_results(model.y_test, model.y_test_pred_ridge)
    shap_values = model.interpret()
    plot_shap_values(shap_values, model.X_test_stack)
    scatter_plot_path = './scatter_plot.png'
    shap_plot_path = './shap_plot.png'
    shap_summary_plot_path = './shap_summary_plot.png'

    # Create the PDF
    pdf_buffer = BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    y_pos = 650
    prediction_time_str = f"Prediction Time: {prediction_time:.2f} seconds"
    p.drawString(100, 750, prediction_time_str)
    for line in results:
        p.drawString(100, y_pos, line)
        y_pos -= 20
        if y_pos < 100:
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


# train model and download prediction result pdf file
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
    data_filled, y = load_data(data_file_path, y_file_path)
    model = Model()
    model.train(data_filled, y)

    # Create the PDF
    pdf_buffer = BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    y_pos = 750
    results = []
    y_test = model.y_test.astype(float) # print model prediction results in pdf
    y_test_pred_ridge = np.array(model.y_test_pred_ridge).astype(float)
    errors = np.abs(y_test.values - y_test_pred_ridge)
    high_error_indices = np.argsort(errors)
    for idx in high_error_indices:
        results.append(f"Instance {idx}:")
        results.append(f"  Actual value: {y_test.iloc[idx]}")
        results.append(f"  Predicted value: {y_test_pred_ridge[idx]}")
        results.append(f"  Error: {errors[idx]}")
        results.append(f"  Feature values: {model.X_test.iloc[idx].to_dict()}")
    max_width = 500
    for line in results:
        wrapped_lines = wrap_text(line, p, max_width)
        for wrapped_line in wrapped_lines:
            p.drawString(50, y_pos, wrapped_line)
            y_pos -= 15
            if y_pos < 100:
                p.showPage()
                y_pos = 750
    p.save()
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name='prediction_results.pdf')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




