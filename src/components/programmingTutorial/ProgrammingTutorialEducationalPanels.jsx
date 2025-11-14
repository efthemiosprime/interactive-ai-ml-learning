import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function ProgrammingTutorialEducationalPanels({ selectedTopic, selectedFramework }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'pytorch-basics' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">PyTorch Basics</h3>
            <p className="text-gray-700 mb-4">
              PyTorch is a deep learning framework developed by Facebook. It's known for its dynamic computation graphs
              and Pythonic interface, making it popular for research and rapid prototyping.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Tensors:</strong> Multi-dimensional arrays (like NumPy arrays but with GPU support)</li>
              <li><strong>Automatic Differentiation:</strong> Automatic computation of gradients</li>
              <li><strong>Dynamic Graphs:</strong> Computation graphs built on-the-fly</li>
              <li><strong>GPU Acceleration:</strong> Seamless CUDA support for faster computation</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Why PyTorch?</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Intuitive Python API</li>
              <li>Great for research and experimentation</li>
              <li>Strong community and ecosystem</li>
              <li>Used by many leading AI research labs</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'tensorflow-basics' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">TensorFlow Basics</h3>
            <p className="text-gray-700 mb-4">
              TensorFlow is Google's open-source deep learning framework. It's known for its production-ready capabilities,
              extensive tooling, and support for both eager execution and graph mode.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Tensors:</strong> Multi-dimensional arrays (similar to PyTorch)</li>
              <li><strong>Eager Execution:</strong> Immediate evaluation (like PyTorch)</li>
              <li><strong>Graph Mode:</strong> Build computation graphs for optimization</li>
              <li><strong>Keras API:</strong> High-level API for building models</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Why TensorFlow?</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Production-ready deployment tools</li>
              <li>TensorFlow Lite for mobile devices</li>
              <li>TensorFlow.js for web deployment</li>
              <li>Strong industry adoption</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'linear-regression' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">
              Linear Regression with {selectedFramework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}
            </h3>
            <p className="text-gray-700 mb-4">
              Linear regression is the simplest machine learning model. It learns a linear relationship between
              input features and a continuous target variable.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">What You'll Learn:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Creating a simple linear model</li>
              <li>Defining loss functions (Mean Squared Error)</li>
              <li>Implementing gradient descent</li>
              <li>Training the model</li>
              <li>Making predictions</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Foundation:</h4>
            <div className="bg-orange-50 rounded-lg p-4 mb-4 border-2 border-orange-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Model:</strong> ŷ = Wx + b</div>
                <div><strong>Loss:</strong> MSE = (1/n) Σ(yᵢ - ŷᵢ)²</div>
                <div><strong>Gradient:</strong> ∂L/∂W = (2/n) Σ(ŷᵢ - yᵢ) × xᵢ</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedTopic === 'logistic-regression' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">
              Logistic Regression with {selectedFramework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}
            </h3>
            <p className="text-gray-700 mb-4">
              Logistic regression is used for binary classification. It outputs probabilities using the sigmoid function.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Sigmoid activation function</li>
              <li>Binary cross-entropy loss</li>
              <li>Probability interpretation</li>
              <li>Decision threshold</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'neural-network' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">
              Neural Networks with {selectedFramework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}
            </h3>
            <p className="text-gray-700 mb-4">
              Build multi-layer neural networks with activation functions, forward pass, and backpropagation.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">What You'll Build:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Multi-layer perceptron (MLP)</li>
              <li>Activation functions (ReLU, Sigmoid, Tanh)</li>
              <li>Forward and backward propagation</li>
              <li>Training with mini-batches</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'pretrained-models' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">Pre-trained Models</h3>
            <p className="text-gray-700 mb-4">
              Use pre-trained models for transfer learning. These models have been trained on large datasets and
              can be fine-tuned for your specific task.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Popular Pre-trained Models:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Image Classification:</strong> ResNet, VGG, EfficientNet</li>
              <li><strong>NLP:</strong> BERT, GPT, T5</li>
              <li><strong>Transfer Learning:</strong> Fine-tuning for your dataset</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'data-loading' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">Data Loading & Preprocessing</h3>
            <p className="text-gray-700 mb-4">
              Learn how to load datasets, preprocess data, and create data loaders for efficient training.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Topics Covered:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Loading datasets (CSV, images, text)</li>
              <li>Data normalization and standardization</li>
              <li>Data augmentation</li>
              <li>Creating batches and data loaders</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'training-loops' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">Training Loops</h3>
            <p className="text-gray-700 mb-4">
              Implement complete training loops with forward pass, loss calculation, backpropagation, and optimization.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Components:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Forward pass</li>
              <li>Loss calculation</li>
              <li>Backward pass (gradient computation)</li>
              <li>Optimizer step (weight update)</li>
              <li>Epoch and batch iteration</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'model-evaluation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-orange-900 mb-3">Model Evaluation</h3>
            <p className="text-gray-700 mb-4">
              Evaluate your models using appropriate metrics for regression and classification tasks.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Evaluation Metrics:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Regression:</strong> MSE, MAE, R²</li>
              <li><strong>Classification:</strong> Accuracy, Precision, Recall, F1-score</li>
              <li>Confusion matrices</li>
              <li>ROC curves</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

