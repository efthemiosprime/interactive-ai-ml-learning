import React from 'react';
import InteractiveClusteringVisualization from './InteractiveClusteringVisualization';
import InteractiveDimensionalityReductionVisualization from './InteractiveDimensionalityReductionVisualization';
import InteractiveAnomalyDetectionVisualization from './InteractiveAnomalyDetectionVisualization';
import InteractiveDistanceMetricsVisualization from './InteractiveDistanceMetricsVisualization';

export default function UnsupervisedLearningVisualization({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'clustering' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Clustering</h3>
          <InteractiveClusteringVisualization />
        </div>
      )}

      {selectedTopic === 'dimensionality-reduction' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Dimensionality Reduction</h3>
          <InteractiveDimensionalityReductionVisualization />
        </div>
      )}

      {selectedTopic === 'anomaly-detection' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Anomaly Detection</h3>
          <InteractiveAnomalyDetectionVisualization />
        </div>
      )}

      {selectedTopic === 'distance-metrics' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Distance Metrics</h3>
          <InteractiveDistanceMetricsVisualization />
        </div>
      )}
    </div>
  );
}

