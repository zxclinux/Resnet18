import React, { useState, useEffect } from 'react';
import { ImageUploader } from './components/ImageUploader';
import Predictions from './components/Predictions';
import { useImageClassifier } from './hooks/useImageClassifier';
import { useHeatmaps } from './hooks/useHeatmaps';
import './App.css';

function App() {
  const [imagePreview, setImagePreview] = useState(null);

  const {
    predictions,
    file: uploadedFile,
    isLoading: isClassifying,
    error: classifyError,
    handleClassify,
  } = useImageClassifier();

  const {
    heatmap,
    generateHeatmap,
    isLoading: isGeneratingHeatmap,
    error: heatmapError,
  } = useHeatmaps();

  useEffect(() => {
    if (predictions && uploadedFile) {
      const topClassIdx = predictions.probabilities.indexOf(
        Math.max(...predictions.probabilities)
      );
      generateHeatmap(uploadedFile, topClassIdx);
    }
  }, [predictions, uploadedFile]);

const handleImageUpload = async (file) => {
  if (!file) return;

  const dataURL = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target.result);
    reader.readAsDataURL(file);
  });

  setImagePreview(dataURL);
  await handleClassify(file);
};


  return (
    <div className="app">
      <h1>CIFAR-10 Classifier with CAM Visualization</h1>

      <ImageUploader onUpload={handleImageUpload} disabled={isClassifying || isGeneratingHeatmap} />

      {(isClassifying || isGeneratingHeatmap) && <div className="loader">Processing...</div>}
      {classifyError && <div className="error">Classification error: {classifyError}</div>}
      {heatmapError && <div className="error">Heatmap error: {heatmapError}</div>}

      {/* Дві картинки поряд */}
      <div className="image-cam-container">
        {imagePreview && (
          <div className="image-box">
            <h3>Original</h3>
            <img src={imagePreview} alt="Uploaded" />
          </div>
        )}
        {heatmap && (
          <div className="image-box">
            <h3>Grad-CAM</h3>
            <img src={heatmap} alt="CAM Heatmap" />
          </div>
        )}
      </div>

      {/* Знизу — лише predictions */}
      {predictions && (
        <div className="predictions-section">
          <Predictions predictions={predictions} />
        </div>
      )}
    </div>
  );
}

export default App;
