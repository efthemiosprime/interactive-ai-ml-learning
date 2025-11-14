import React from 'react';
import InteractiveArchitectureVisualization from './InteractiveArchitectureVisualization';
import InteractiveForwardPassVisualization from './InteractiveForwardPassVisualization';
import InteractiveBackpropagationVisualization from './InteractiveBackpropagationVisualization';
import InteractiveActivationFunctionsVisualization from './InteractiveActivationFunctionsVisualization';
import InteractiveTransformersVisualization from './InteractiveTransformersVisualization';
import InteractiveTrainingVisualization from './InteractiveTrainingVisualization';

export default function NeuralNetworksVisualization({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'architecture' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Neural Network Architecture</h3>
          <InteractiveArchitectureVisualization />
        </div>
      )}

      {selectedTopic === 'forward-pass' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Forward Pass</h3>
          <InteractiveForwardPassVisualization />
        </div>
      )}

      {selectedTopic === 'backpropagation' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Backpropagation</h3>
          <InteractiveBackpropagationVisualization />
        </div>
      )}

      {selectedTopic === 'activation-functions' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Activation Functions</h3>
          <InteractiveActivationFunctionsVisualization />
        </div>
      )}

      {selectedTopic === 'transformers' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Transformers & Attention</h3>
          <InteractiveTransformersVisualization />
        </div>
      )}

      {selectedTopic === 'training' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Training Process</h3>
          <InteractiveTrainingVisualization />
        </div>
      )}
    </div>
  );
}

