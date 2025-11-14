import React from 'react';
import * as stats from '../../utils/probabilityStatistics';

export default function ProbabilityStatisticsControls({ selectedTopic, setSelectedTopic, dataSet, setDataSet }) {
  const topics = [
    { id: 'descriptive', label: 'Descriptive Statistics' },
    { id: 'covariance', label: 'Covariance & Correlation' },
    { id: 'conditional-probability', label: 'Conditional Probability' },
    { id: 'bayes', label: "Bayes' Theorem" },
    { id: 'distributions', label: 'Probability Distributions' }
  ];

  const handleDataSetChange = (value) => {
    const values = value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    setDataSet(values.length > 0 ? values : [0]);
  };

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
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Data Set Input */}
      {(selectedTopic === 'descriptive' || selectedTopic === 'covariance') && (
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Data Set (comma-separated)
          </label>
          <input
            type="text"
            value={dataSet.join(', ')}
            onChange={(e) => handleDataSetChange(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            placeholder="1, 2, 3, 4, 5"
          />
          <p className="text-xs text-gray-500 mt-1">
            {dataSet.length} values
          </p>
        </div>
      )}

      {/* Statistics Display */}
      {selectedTopic === 'descriptive' && dataSet.length > 0 && (
        <div className="mb-6 bg-blue-50 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">Quick Stats</h3>
          <div className="text-sm space-y-1 text-blue-800">
            <div>Mean: {stats.calculateDescriptiveStats(dataSet).mean.toFixed(2)}</div>
            <div>Std Dev: {stats.calculateDescriptiveStats(dataSet).standardDeviation.toFixed(2)}</div>
            <div>Variance: {stats.calculateDescriptiveStats(dataSet).variance.toFixed(2)}</div>
          </div>
        </div>
      )}

      {/* Info Panel */}
      <div className="bg-blue-50 rounded-lg p-4">
        <h3 className="font-semibold text-blue-900 mb-2">ML Application</h3>
        <p className="text-sm text-blue-800">
          {selectedTopic === 'descriptive' && 
            'Descriptive statistics summarize data characteristics, essential for data preprocessing.'}
          {selectedTopic === 'covariance' && 
            'Covariance measures feature relationships, used in feature selection and PCA.'}
          {selectedTopic === 'conditional-probability' && 
            'Conditional probability models dependencies between variables in probabilistic models.'}
          {selectedTopic === 'bayes' && 
            "Bayes' theorem is fundamental to Naive Bayes classifiers and Bayesian inference."}
          {selectedTopic === 'distributions' && 
            'Probability distributions model uncertainty and are used in generative models.'}
        </p>
      </div>
    </div>
  );
}

