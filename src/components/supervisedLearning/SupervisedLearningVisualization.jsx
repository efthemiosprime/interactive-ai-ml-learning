import React from 'react';
import InteractiveFoundationsVisualization from './InteractiveFoundationsVisualization';
import InteractiveKeyConceptsVisualization from './InteractiveKeyConceptsVisualization';
import InteractiveLossFunctionsVisualization from './InteractiveLossFunctionsVisualization';
import InteractiveModelEvaluationVisualization from './InteractiveModelEvaluationVisualization';
import InteractiveBiasVarianceVisualization from './InteractiveBiasVarianceVisualization';
import InteractiveRegularizationVisualization from './InteractiveRegularizationVisualization';

export default function SupervisedLearningVisualization({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'foundations' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Mathematical Foundations</h3>
          <InteractiveFoundationsVisualization />
        </div>
      )}

      {selectedTopic === 'key-concepts' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Key Concepts & Terms</h3>
          <InteractiveKeyConceptsVisualization />
        </div>
      )}

      {selectedTopic === 'loss-functions' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Loss Functions</h3>
          <InteractiveLossFunctionsVisualization />
        </div>
      )}

      {selectedTopic === 'model-evaluation' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Model Evaluation Metrics</h3>
          <InteractiveModelEvaluationVisualization />
        </div>
      )}

      {selectedTopic === 'bias-variance' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Bias-Variance Tradeoff</h3>
          <InteractiveBiasVarianceVisualization />
        </div>
      )}

      {selectedTopic === 'regularization' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Regularization (L1/L2)</h3>
          <InteractiveRegularizationVisualization />
        </div>
      )}
    </div>
  );
}

