import React from 'react';

export default function SupervisedLearningControls({ 
  selectedTopic, 
  setSelectedTopic
}) {
  const topics = [
    { id: 'foundations', label: 'Mathematical Foundations' },
    { id: 'key-concepts', label: 'Key Concepts & Terms' },
    { id: 'loss-functions', label: 'Loss Functions' },
    { id: 'model-evaluation', label: 'Model Evaluation Metrics' },
    { id: 'bias-variance', label: 'Bias-Variance Tradeoff' },
    { id: 'regularization', label: 'Regularization (L1/L2)' }
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
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Info Panel */}
      <div className="bg-green-50 rounded-lg p-4">
        <h3 className="font-semibold text-green-900 mb-2">ML Application</h3>
        <p className="text-sm text-green-800">
          {selectedTopic === 'foundations' && 
            'How Linear Algebra, Calculus, and Probability form the mathematical foundation of supervised learning.'}
          {selectedTopic === 'key-concepts' && 
            'Understanding fundamental supervised learning concepts: regression, classification, and algorithms.'}
          {selectedTopic === 'loss-functions' && 
            'Loss functions measure prediction error and guide model optimization.'}
          {selectedTopic === 'model-evaluation' && 
            'Evaluation metrics assess model performance on unseen data.'}
          {selectedTopic === 'bias-variance' && 
            'Understanding the tradeoff between model complexity and generalization.'}
          {selectedTopic === 'regularization' && 
            'Regularization prevents overfitting by penalizing complex models.'}
        </p>
      </div>
    </div>
  );
}

