import React from 'react';

export default function NeuralNetworksControls({ 
  selectedTopic, 
  setSelectedTopic
}) {
  const topics = [
    { id: 'architecture', label: 'Neural Network Architecture' },
    { id: 'forward-pass', label: 'Forward Pass' },
    { id: 'backpropagation', label: 'Backpropagation' },
    { id: 'activation-functions', label: 'Activation Functions' },
    { id: 'transformers', label: 'Transformers & Attention' },
    { id: 'training', label: 'Training Process' }
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
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Info Panel */}
      <div className="bg-violet-50 rounded-lg p-4">
        <h3 className="font-semibold text-violet-900 mb-2">ML Application</h3>
        <p className="text-sm text-violet-800">
          {selectedTopic === 'architecture' && 
            'Neural networks are the foundation of deep learning and modern AI.'}
          {selectedTopic === 'forward-pass' && 
            'Forward pass computes predictions by propagating data through layers.'}
          {selectedTopic === 'backpropagation' && 
            'Backpropagation calculates gradients to update network weights.'}
          {selectedTopic === 'activation-functions' && 
            'Activation functions introduce non-linearity into neural networks.'}
          {selectedTopic === 'transformers' && 
            'Transformers use attention mechanisms and power modern LLMs.'}
          {selectedTopic === 'training' && 
            'Training optimizes network weights to minimize prediction error.'}
        </p>
      </div>
    </div>
  );
}

