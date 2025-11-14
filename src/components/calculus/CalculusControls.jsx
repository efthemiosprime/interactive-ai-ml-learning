import React from 'react';

export default function CalculusControls({ selectedTopic, setSelectedTopic, functionType, setFunctionType }) {
  const topics = [
    { id: 'derivatives', label: 'Derivatives' },
    { id: 'partial-derivatives', label: 'Partial Derivatives' },
    { id: 'gradients', label: 'Gradients' },
    { id: 'chain-rule', label: 'Chain Rule' },
    { id: 'backpropagation', label: 'Backpropagation' }
  ];

  const functionTypes = [
    { id: 'quadratic', label: 'Quadratic (x²)' },
    { id: 'cubic', label: 'Cubic (x³)' },
    { id: 'sine', label: 'Sine (sin(x))' },
    { id: 'exponential', label: 'Exponential (eˣ)' }
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
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Function Type Selector */}
      {selectedTopic !== 'backpropagation' && (
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Function Type
          </label>
          <select
            value={functionType}
            onChange={(e) => setFunctionType(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            {functionTypes.map(type => (
              <option key={type.id} value={type.id}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Info Panel */}
      <div className="bg-purple-50 rounded-lg p-4">
        <h3 className="font-semibold text-purple-900 mb-2">ML Application</h3>
        <p className="text-sm text-purple-800">
          {selectedTopic === 'derivatives' && 
            'Derivatives measure how a function changes, essential for optimization.'}
          {selectedTopic === 'partial-derivatives' && 
            'Partial derivatives show how a function changes with respect to one variable.'}
          {selectedTopic === 'gradients' && 
            'Gradients point in the direction of steepest ascent, used in gradient descent.'}
          {selectedTopic === 'chain-rule' && 
            'Chain rule enables backpropagation by computing derivatives of composite functions.'}
          {selectedTopic === 'backpropagation' && 
            'Backpropagation uses chain rule to compute gradients for neural network training.'}
        </p>
      </div>
    </div>
  );
}

