import React from 'react';
import { CIFAR10_CLASSES } from '../constants/cifar10Classes';

const Predictions = ({ predictions}) => {
  return (
    <div className="predictions">
        <h2>Class:</h2>
         <li>{predictions.class} (ID: {predictions.class_idx})</li>
      <h2>Predictions</h2>
      <ul>
        {predictions.probabilities.map((prob, idx) => (
          <li key={idx}>
            <span>{CIFAR10_CLASSES[idx]}: {(prob * 100).toFixed(2)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Predictions;