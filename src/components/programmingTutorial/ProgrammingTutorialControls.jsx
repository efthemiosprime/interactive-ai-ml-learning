import React from 'react';

export default function ProgrammingTutorialControls({ 
  selectedTopic, 
  setSelectedTopic,
  selectedFramework,
  setSelectedFramework
}) {
  const topics = [
    { id: 'pytorch-basics', label: 'PyTorch Basics', framework: 'pytorch' },
    { id: 'tensorflow-basics', label: 'TensorFlow Basics', framework: 'tensorflow' },
    { id: 'linear-regression', label: 'Linear Regression', framework: 'both' },
    { id: 'logistic-regression', label: 'Logistic Regression', framework: 'both' },
    { id: 'neural-network', label: 'Neural Network', framework: 'both' },
    { id: 'pretrained-models', label: 'Pre-trained Models', framework: 'both' },
    { id: 'data-loading', label: 'Data Loading & Preprocessing', framework: 'both' },
    { id: 'training-loops', label: 'Training Loops', framework: 'both' },
    { id: 'model-evaluation', label: 'Model Evaluation', framework: 'both' }
  ];

  const filteredTopics = topics.filter(topic => 
    topic.framework === 'both' || 
    topic.framework === selectedFramework ||
    (selectedFramework === 'pytorch' && topic.id === 'pytorch-basics') ||
    (selectedFramework === 'tensorflow' && topic.id === 'tensorflow-basics')
  );

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Controls</h2>

      {/* Framework Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Framework
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setSelectedFramework('pytorch')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedFramework === 'pytorch'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            PyTorch
          </button>
          <button
            onClick={() => setSelectedFramework('tensorflow')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedFramework === 'tensorflow'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            TensorFlow
          </button>
        </div>
      </div>

      {/* Topic Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Topic
        </label>
        <select
          value={selectedTopic}
          onChange={(e) => setSelectedTopic(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
        >
          {filteredTopics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Info Panel */}
      <div className="bg-orange-50 rounded-lg p-4">
        <h3 className="font-semibold text-orange-900 mb-2">About This Topic</h3>
        <p className="text-sm text-orange-800">
          {selectedTopic === 'pytorch-basics' && 
            'Learn the fundamentals of PyTorch: tensors, operations, and automatic differentiation.'}
          {selectedTopic === 'tensorflow-basics' && 
            'Learn the fundamentals of TensorFlow: tensors, operations, and eager execution.'}
          {selectedTopic === 'linear-regression' && 
            'Build a linear regression model from scratch using the selected framework.'}
          {selectedTopic === 'logistic-regression' && 
            'Build a logistic regression model for binary classification.'}
          {selectedTopic === 'neural-network' && 
            'Create a multi-layer neural network with activation functions.'}
          {selectedTopic === 'pretrained-models' && 
            'Use pre-trained models for transfer learning and inference.'}
          {selectedTopic === 'data-loading' && 
            'Load and preprocess datasets for machine learning.'}
          {selectedTopic === 'training-loops' && 
            'Implement training loops with loss calculation and optimization.'}
          {selectedTopic === 'model-evaluation' && 
            'Evaluate model performance using various metrics.'}
        </p>
      </div>
    </div>
  );
}

