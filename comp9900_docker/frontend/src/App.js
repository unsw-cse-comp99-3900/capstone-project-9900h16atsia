import React, { useState } from 'react';
import './App.css';

function App() {
  const [dataFile, setDataFile] = useState(null);
  const [yFile, setYFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleDataFileChange = (event) => {
    setDataFile(event.target.files[0]);
    console.log('Data file selected:', event.target.files[0]);
  };

  const handleYFileChange = (event) => {
    setYFile(event.target.files[0]);
    console.log('Y file selected:', event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!dataFile || !yFile) {
      console.log('Please select both data and Y files.');
      return;
    }

    console.log('Uploading files...');
    const formData = new FormData();
    formData.append('data_file', dataFile);
    formData.append('y_file', yFile);

    try {
      const response = await fetch(process.env.REACT_APP_BACKEND_URL + '/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPrediction(data.prediction);
      console.log('Prediction received:', data.prediction);
    } catch (error) {
      console.error('Error during upload:', error);
    }
  };

  return (
    <div className="App">
      <div className="animated-bg"></div>
      <h1>Predicting Contaminant Degradation in Groundwater using Machine Learning</h1>
      <div className="container">
        <div className="function-container">
          <div className="function-box">
            <div className="function-title">Degradation Prediction</div>
            <button className="btn" onClick={handleUpload}>Predict Degradation Rates</button>
          </div>
          <div className="function-box">
            <div className="function-title">Upload Data</div>
            <label htmlFor="data-file-upload" className="file-label">Choose Data File</label>
            <input id="data-file-upload" className="file-input" type="file" onChange={handleDataFileChange} />
            <label htmlFor="y-file-upload" className="file-label">Choose Y File</label>
            <input id="y-file-upload" className="file-input" type="file" onChange={handleYFileChange} />
            <button className="btn" onClick={handleUpload}>Upload Dataset</button>
          </div>
        </div>
      </div>
      {prediction && (
        <div className="result">
          <h2>Prediction Result</h2>
          <ul>
            {prediction.map((pred, index) => (
              <li key={index}>{pred}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;








