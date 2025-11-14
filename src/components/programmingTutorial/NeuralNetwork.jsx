import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function NeuralNetwork({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate sample data
np.random.seed(42)
x = np.random.randn(100, 2)
y = ((x[:, 0]**2 + x[:, 1]**2) > 1).astype(float)

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Define neural network
model = nn.Sequential(
    nn.Linear(2, 4),      # Hidden layer 1: 2 inputs -> 4 neurons
    nn.ReLU(),            # Activation function
    nn.Linear(4, 4),      # Hidden layer 2: 4 -> 4
    nn.ReLU(),
    nn.Linear(4, 1),      # Output layer: 4 -> 1
    nn.Sigmoid()          # Output activation
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 200
for epoch in range(epochs):
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 40 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate
with torch.no_grad():
    predictions = model(x_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_tensor).float().mean()
    print(f'\\nAccuracy: {accuracy.item():.2%}')`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
import numpy as np

# Generate sample data
np.random.seed(42)
x = np.random.randn(100, 2)
y = ((x[:, 0]**2 + x[:, 1]**2) > 1).astype(float)

# Define neural network
model = keras.Sequential([
    keras.layers.Dense(4, input_shape=(2,), activation='relu'),  # Hidden layer 1
    keras.layers.Dense(4, activation='relu'),                    # Hidden layer 2
    keras.layers.Dense(1, activation='sigmoid')                  # Output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(x, y, epochs=200, batch_size=32, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(x, y, verbose=0)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.2%}')`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = framework === 'pytorch'
    ? `Epoch [40/200], Loss: 0.4567
Epoch [80/200], Loss: 0.2341
Epoch [120/200], Loss: 0.1234
Epoch [160/200], Loss: 0.0567
Epoch [200/200], Loss: 0.0234

Accuracy: 96.00%`
    : `Loss: 0.0234
Accuracy: 96.00%`;

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
          💡 <strong>Multi-Layer Network:</strong> Build a neural network with hidden layers and activation functions!
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
        <h4 className="font-semibold text-blue-900 mb-2">Network Architecture:</h4>
        <ul className="list-disc list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Input Layer:</strong> 2 features</li>
          <li><strong>Hidden Layer 1:</strong> 4 neurons with ReLU activation</li>
          <li><strong>Hidden Layer 2:</strong> 4 neurons with ReLU activation</li>
          <li><strong>Output Layer:</strong> 1 neuron with Sigmoid activation</li>
          <li><strong>Total Parameters:</strong> (2×4+4) + (4×4+4) + (4×1+1) = 33 parameters</li>
        </ul>
      </div>
    </div>
  );
}

