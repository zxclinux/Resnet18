import React from 'react';

const Heatmaps = ({ heatmap }) => {
  return (
    <div className="heatmaps">
      <h2>Class Activation Map</h2>
      <img src={heatmap} alt="CAM Heatmap" />
    </div>
  );
};

export default Heatmaps;