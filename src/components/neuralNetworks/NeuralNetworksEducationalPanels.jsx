import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function NeuralNetworksEducationalPanels({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'architecture' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Neural Network Architecture</h3>
            <p className="text-gray-700 mb-4">
              Neural networks are composed of interconnected layers of neurons (nodes) that process information. 
              Each connection has a weight that gets adjusted during training.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Key Components:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Input Layer:</strong> Receives input data (features)</li>
              <li><strong>Hidden Layers:</strong> Process information through weighted connections</li>
              <li><strong>Output Layer:</strong> Produces final predictions</li>
              <li><strong>Weights:</strong> Parameters that get learned during training</li>
              <li><strong>Biases:</strong> Additional parameters that shift activation functions</li>
            </ul>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Neuron Output:</strong> activation(Σ(weight × input) + bias)</div>
                <div className="text-xs text-gray-600 mt-2">
                  Each neuron computes a weighted sum of inputs, adds bias, then applies activation function
                </div>
              </div>
            </div>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="architecture" />
        </div>
      )}

      {selectedTopic === 'forward-pass' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Forward Pass</h3>
            <p className="text-gray-700 mb-4">
              The forward pass propagates input data through the network layers to produce predictions. 
              This is how neural networks make predictions.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Process:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li>Input values enter the input layer</li>
              <li>Each neuron computes: weighted sum of inputs + bias</li>
              <li>Activation function is applied to the sum</li>
              <li>Result becomes input to next layer</li>
              <li>Process repeats until output layer produces final prediction</li>
            </ol>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Layer Computation:</strong> a^(l) = activation(W^(l) × a^(l-1) + b^(l))</div>
                <div className="text-xs text-gray-600 mt-2">
                  Where W is weight matrix, b is bias vector, a is activation
                </div>
              </div>
            </div>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="forward-pass" />
        </div>
      )}

      {selectedTopic === 'backpropagation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Backpropagation</h3>
            <p className="text-gray-700 mb-4">
              Backpropagation is the algorithm that trains neural networks by computing gradients 
              and updating weights to minimize prediction error.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Algorithm Steps:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li>Forward pass: Compute predictions</li>
              <li>Calculate loss: Compare predictions to targets</li>
              <li>Compute output layer error</li>
              <li>Propagate error backward through layers (chain rule)</li>
              <li>Calculate gradients for each weight and bias</li>
              <li>Update weights: w = w - α × gradient</li>
            </ol>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Chain Rule:</strong> ∂L/∂w = (∂L/∂y) × (∂y/∂z) × (∂z/∂w)</div>
                <div className="text-xs text-gray-600 mt-2">
                  Gradients flow backward, multiplied at each layer
                </div>
              </div>
            </div>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="backpropagation" />
        </div>
      )}

      {selectedTopic === 'activation-functions' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Activation Functions</h3>
            <p className="text-gray-700 mb-4">
              Activation functions introduce non-linearity into neural networks, enabling them 
              to learn complex patterns. Without them, neural networks would just be linear transformations.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Common Activation Functions:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>ReLU:</strong> f(x) = max(0, x). Most popular, solves vanishing gradient problem</li>
              <li><strong>Sigmoid:</strong> f(x) = 1/(1+e^(-x)). Outputs 0-1, used in binary classification</li>
              <li><strong>Tanh:</strong> f(x) = tanh(x). Outputs -1 to 1, zero-centered</li>
              <li><strong>Linear:</strong> f(x) = x. No transformation, rarely used in hidden layers</li>
            </ul>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Why Non-Linear:</strong> Multiple linear layers = single linear layer</div>
                <div className="text-xs text-gray-600 mt-2">
                  Non-linearity enables learning complex, non-linear relationships
                </div>
              </div>
            </div>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="activation-functions" />
        </div>
      )}

      {selectedTopic === 'transformers' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Transformers & Attention</h3>
            <p className="text-gray-700 mb-4">
              Transformers revolutionized NLP and power modern LLMs. They use self-attention mechanisms 
              to process sequences in parallel and capture long-range dependencies.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Self-Attention:</strong> Each token attends to all other tokens in the sequence</li>
              <li><strong>Query, Key, Value:</strong> QKV mechanism computes attention scores</li>
              <li><strong>Multi-Head Attention:</strong> Multiple attention heads capture different relationships</li>
              <li><strong>Positional Encoding:</strong> Adds position information to tokens</li>
              <li><strong>Encoder-Decoder:</strong> Encoder processes input, decoder generates output</li>
            </ul>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Attention:</strong> Attention(Q, K, V) = softmax(QK^T / √d_k) × V</div>
                <div className="text-xs text-gray-600 mt-2">
                  Scaled dot-product attention: computes relevance and weighted sum
                </div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Why Transformers?</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Parallel processing (faster than RNNs)</li>
              <li>Captures long-range dependencies</li>
              <li>Scalable to very large models</li>
              <li>Foundation for GPT, BERT, and modern LLMs</li>
            </ul>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="transformers" />
        </div>
      )}

      {selectedTopic === 'training' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-violet-900 mb-3">Training Process</h3>
            <p className="text-gray-700 mb-4">
              Training neural networks involves iteratively adjusting weights to minimize prediction error. 
              This is an optimization problem solved using gradient descent.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Training Loop:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li>Initialize weights randomly</li>
              <li>Forward pass: Make predictions on training data</li>
              <li>Calculate loss: Measure prediction error</li>
              <li>Backpropagation: Compute gradients</li>
              <li>Update weights: w = w - α × ∇L</li>
              <li>Repeat for multiple epochs until convergence</li>
            </ol>

            <div className="bg-violet-50 rounded-lg p-4 mb-4 border-2 border-violet-200">
              <div className="font-mono text-sm">
                <div><strong>Gradient Descent:</strong> θ := θ - α × (∂L/∂θ)</div>
                <div className="text-xs text-gray-600 mt-2">
                  α = learning rate, controls step size in optimization
                </div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Epoch:</strong> One complete pass through training data</li>
              <li><strong>Batch:</strong> Subset of data processed together</li>
              <li><strong>Learning Rate:</strong> Controls how much weights change</li>
              <li><strong>Loss Function:</strong> Measures prediction error (MSE, Cross-entropy)</li>
              <li><strong>Overfitting:</strong> Model memorizes training data, doesn't generalize</li>
            </ul>
          </div>
          
          <MLUseCasesPanel domain="neural-networks" operationType="training" />
        </div>
      )}
    </div>
  );
}

