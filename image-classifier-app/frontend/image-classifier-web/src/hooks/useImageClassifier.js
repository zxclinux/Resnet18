import { useState } from 'react';
import { classifyImage } from '../api/cifar10Api';

export const useImageClassifier = () => {
  const [predictions, setPredictions] = useState(null);
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClassify = async (file) => {
    setFile(file);
    setIsLoading(true);
    setError(null);

    try {
      const data = await classifyImage(file);
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return { predictions, file, isLoading, error, handleClassify };
};
