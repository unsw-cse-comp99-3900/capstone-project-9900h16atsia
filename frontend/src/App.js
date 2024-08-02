import React, { useState } from 'react';
import './App.css';
import Button from '@material-ui/core/Button';

function App () {
  const [dataFile, setDataFile] = useState(null);
  const [yFile, setYFile] = useState(null);

  const handleDataFileChange = (event) => {
    setDataFile(event.target.files[0]);
    console.log('Data file selected:', event.target.files[0]);
  };

  const handleYFileChange = (event) => {
    setYFile(event.target.files[0]);
    console.log('Y file selected:', event.target.files[0]);
  };

  const handleUploadDataset = async () => {
    if (!dataFile || !yFile) {
      console.log('Please select both data and Y files.');
      return;
    }

    console.log('Uploading files...');
    const formData = new FormData();
    formData.append('data_file', dataFile);
    formData.append('y_file', yFile);

    try {
      const response = await fetch(process.env.REACT_APP_BACKEND_URL + '/upload_dataset', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Data file uploaded successfully');
      } else {
        console.error('Failed to upload data file');
      }
    } catch (error) {
      console.error('Error during upload:', error);
    }
  };

  const handlePrediction = async () => {
    if (!dataFile || !yFile) {
      console.log('select both data and Y files.');
      return;
    }

    console.log('Uploading files...');
    const formData = new FormData();
    formData.append('data_file', dataFile);
    formData.append('y_file', yFile);
    const backendUrl = process.env.REACT_APP_BACKEND_URL + '/predict';
    console.log('Request URL:', backendUrl);
    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'prediction_results.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      console.log('Download initiated');
    } catch (error) {
      console.error('Error during upload:', error);
    }
  };

    const handleAnalysis = async () => {
    if (!dataFile || !yFile) {
      console.log('Please select both data and Y files.');
      return;
    }

    console.log('Uploading files...');
    const formData = new FormData();
    formData.append('data_file', dataFile);
    formData.append('y_file', yFile);
    const backendUrl = process.env.REACT_APP_BACKEND_URL + '/analysis';
    console.log('Request URL:', backendUrl);

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'analyse_results.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      console.log('Download initiated');
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
            <Button onClick={handlePrediction} variant="contained" color="primary" >Predict Degradation Rates</Button>
          </div>
          <div className="function-box right-box ">
            <div className="function-title">Upload Data</div>
            <div className='flex justify-between' >
              <input id="data-file-upload" className="file-input" type="file" onChange={handleDataFileChange} />
              <label htmlFor="data-file-upload">
                <Button variant="contained" color="primary" component="span">Choose Data File</Button>
              </label>
              <input id="y-file-upload" className="file-input" type="file" onChange={handleYFileChange} />
              <label htmlFor="y-file-upload">
                <Button variant="contained" color="primary" component="span">Choose Y File</Button>
              </label>
              <Button onClick={handleUploadDataset} variant="contained" color="primary" >Upload Dataset</Button>

            </div>
          </div>
        </div>
        <div className='function-container mt bottom-info flex-column' >
          <div className='flex item-center w-full label-item' >
            <div className="label-title">Monitor model performance:</div>
            <Button onClick={handleAnalysis} variant="contained" color="secondary" >response rate</Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;








