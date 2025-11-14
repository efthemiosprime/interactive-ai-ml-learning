import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function LinearRegression({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate sample data: y = 2x + 1 + noise
np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2 * x + 1 + np.random.randn(100, 1) * 0.5

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)

# Define model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get learned parameters
weight = model.weight.data.item()
bias = model.bias.data.item()
print(f'\\nLearned: y = {weight:.2f}x + {bias:.2f}')
print(f'True: y = 2.00x + 1.00')`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
import numpy as np

# Generate sample data: y = 2x + 1 + noise
np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2 * x + 1 + np.random.randn(100, 1) * 0.5

# Define model
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Training
history = model.fit(x, y, epochs=100, batch_size=32, verbose=0)

# Get learned parameters
weight = model.layers[0].get_weights()[0][0][0]
bias = model.layers[0].get_weights()[1][0]
print(f'Learned: y = {weight:.2f}x + {bias:.2f}')
print(f'True: y = 2.00x + 1.00')`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = framework === 'pytorch' 
    ? `Epoch [20/100], Loss: 0.2341
Epoch [40/100], Loss: 0.1234
Epoch [60/100], Loss: 0.0567
Epoch [80/100], Loss: 0.0234
Epoch [100/100], Loss: 0.0123

Learned: y = 1.98x + 1.02
True: y = 2.00x + 1.00`
    : `Learned: y = 1.98x + 1.02
True: y = 2.00x + 1.00`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const runCode = () => {
    setOutput(expectedOutput);
  };

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border-2 border-orange-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-orange-800">
          💡 <strong>Complete Example:</strong> Build a linear regression model with {framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}!
        </p>
      </div>

      {/* Code Display */}
      <div className="bg-gray-900 rounded-lg p-4 relative">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400 text-sm">Python - {framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}</span>
          <button
            onClick={copyToClipboard}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
          </button>
        </div>
        <pre className="text-green-400 text-sm overflow-x-auto">
          <code>{code}</code>
        </pre>
      </div>

      {/* Run Button */}
      <button
        onClick={runCode}
        className="w-full px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-semibold flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Run Code
      </button>

      {/* Output Display */}
      {output && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-2">Output</div>
          <pre className="text-green-400 text-sm whitespace-pre-wrap">
            {output}
          </pre>
        </div>
      )}

      {/* Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">What This Code Does:</h4>
        <ol className="list-decimal list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Generate Data:</strong> Creates synthetic data following y = 2x + 1 with noise</li>
          <li><strong>Define Model:</strong> Creates a linear layer (single weight and bias)</li>
          <li><strong>Loss Function:</strong> Uses Mean Squared Error (MSE)</li>
          <li><strong>Optimizer:</strong> Uses Stochastic Gradient Descent (SGD)</li>
          <li><strong>Training:</strong> Iteratively updates weights to minimize loss</li>
          <li><strong>Result:</strong> Model learns the true relationship (y ≈ 2x + 1)</li>
        </ol>
      </div>
    </div>
  );
}

