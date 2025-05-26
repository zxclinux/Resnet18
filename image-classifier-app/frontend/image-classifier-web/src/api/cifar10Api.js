export const classifyImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/api/v1/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

export const HeatmapImage = async (file, classIdx) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('class_idx', classIdx.toString());

  const response = await fetch('http://localhost:8000/api/v1/heatmaps-cam', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to generate heatmap: ${response.status} — ${errorText}`);
  }

  return await response.blob();
};

