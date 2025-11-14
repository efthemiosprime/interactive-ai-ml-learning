import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function CalculusEducationalPanels({ selectedTopic, functionType }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'derivatives' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Derivatives in Machine Learning</h3>
            <p className="text-gray-700 mb-4">
              Derivatives measure the rate of change of a function. In ML, they're used to find 
              optimal parameters by minimizing loss functions.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Slope:</strong> The derivative at a point gives the slope of the tangent line</li>
              <li><strong>Optimization:</strong> Finding where derivative equals zero locates minima/maxima</li>
              <li><strong>Gradient Descent:</strong> Uses derivatives to iteratively minimize loss functions</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="derivatives" />
        </div>
      )}

      {selectedTopic === 'partial-derivatives' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Partial Derivatives</h3>
            <p className="text-gray-700 mb-4">
              Partial derivatives measure how a multivariable function changes with respect to one 
              variable while keeping others constant. Essential for understanding gradients.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">ML Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Multi-parameter Optimization:</strong> Each parameter has its own partial derivative</li>
              <li><strong>Gradient Computation:</strong> Gradients are vectors of partial derivatives</li>
              <li><strong>Feature Importance:</strong> Partial derivatives show feature sensitivity</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'gradients' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Gradients and Gradient Descent</h3>
            <p className="text-gray-700 mb-4">
              The gradient is a vector pointing in the direction of steepest ascent. Gradient descent 
              moves in the opposite direction to minimize loss functions.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Gradient Descent Algorithm:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li>Initialize parameters randomly</li>
              <li>Compute gradient of loss function</li>
              <li>Update parameters: θ = θ - α∇L(θ)</li>
              <li>Repeat until convergence</li>
            </ol>
            <p className="text-gray-700 mb-4">
              Where α is the learning rate controlling step size.
            </p>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="gradients" />
        </div>
      )}

      {selectedTopic === 'chain-rule' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Chain Rule</h3>
            <p className="text-gray-700 mb-4">
              The chain rule enables computing derivatives of composite functions. This is the 
              mathematical foundation of backpropagation in neural networks.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Why It Matters:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Neural Networks:</strong> Composed of multiple layers (composite functions)</li>
              <li><strong>Backpropagation:</strong> Uses chain rule to compute gradients layer by layer</li>
              <li><strong>Efficiency:</strong> Allows efficient gradient computation for deep networks</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'backpropagation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Backpropagation</h3>
            <p className="text-gray-700 mb-4">
              Backpropagation is the algorithm used to train neural networks. It efficiently computes 
              gradients for all weights using the chain rule.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">How It Works:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Forward Pass:</strong> Compute network output for given input</li>
              <li><strong>Compute Loss:</strong> Compare output to target</li>
              <li><strong>Backward Pass:</strong> Propagate error backward using chain rule</li>
              <li><strong>Update Weights:</strong> Adjust weights using computed gradients</li>
            </ol>
            <p className="text-gray-700 mb-4">
              This process is repeated for many iterations until the network learns.
            </p>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="backpropagation" />
        </div>
      )}
    </div>
  );
}

