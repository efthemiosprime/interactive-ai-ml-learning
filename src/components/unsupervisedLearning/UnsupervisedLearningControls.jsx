import React from 'react';

export default function UnsupervisedLearningControls({ 
  selectedTopic, 
  setSelectedTopic
}) {
  const topics = [
    { id: 'clustering', label: 'Clustering' },
    { id: 'dimensionality-reduction', label: 'Dimensionality Reduction' },
    { id: 'anomaly-detection', label: 'Anomaly Detection' },
    { id: 'distance-metrics', label: 'Distance Metrics' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Controls</h2>

      {/* Topic Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Topic
        </label>
        <select
          value={selectedTopic}
          onChange={(e) => setSelectedTopic(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Info Panel */}
      <div className="bg-teal-50 rounded-lg p-4">
        <h3 className="font-semibold text-teal-900 mb-2">ML Application</h3>
        <p className="text-sm text-teal-800">
          {selectedTopic === 'clustering' && 
            'Clustering groups similar data points together without labels.'}
          {selectedTopic === 'dimensionality-reduction' && 
            'Dimensionality reduction reduces features while preserving important information.'}
          {selectedTopic === 'anomaly-detection' && 
            'Anomaly detection identifies unusual patterns or outliers in data.'}
          {selectedTopic === 'distance-metrics' && 
            'Distance metrics measure similarity between data points.'}
        </p>
      </div>
    </div>
  );
}

