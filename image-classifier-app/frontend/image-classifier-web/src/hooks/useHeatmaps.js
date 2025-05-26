import { useState } from 'react';
import {HeatmapImage} from "../api/cifar10Api";

export const useHeatmaps = () => {
  const [heatmap, setHeatmap] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateHeatmap = async (imageBlob, classIdx) => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await HeatmapImage(imageBlob, classIdx);
      setHeatmap(URL.createObjectURL(data));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return { heatmap, generateHeatmap, isLoading, error };
};