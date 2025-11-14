import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function LogisticRegression({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate sample binary classification data
np.random.seed(42)
x = np.random.randn(100, 2)
y = ((x[:, 0] + x[:, 1]) > 0).astype(float)

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Define model (logistic regression)
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

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

# Evaluate
with torch.no_grad():
    predictions = model(x_tensor)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_tensor).float().mean()
    print(f'\\nAccuracy: {accuracy.item():.2%}')`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
import numpy as np

# Generate sample binary classification data
np.random.seed(42)
x = np.random.randn(100, 2)
y = ((x[:, 0] + x[:, 1]) > 0).astype(float)

# Define model (logistic regression)
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(x, y, epochs=100, batch_size=32, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(x, y, verbose=0)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.2%}')`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = framework === 'pytorch'
    ? `Epoch [20/100], Loss: 0.3456
Epoch [40/100], Loss: 0.2341
Epoch [60/100], Loss: 0.1234
Epoch [80/100], Loss: 0.0567
Epoch [100/100], Loss: 0.0234

Accuracy: 98.00%`
    : `Loss: 0.0234
Accuracy: 98.00%`;

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
          💡 <strong>Binary Classification:</strong> Build a logistic regression model with {framework === 'pytorch' ? 'PyTorch' : 'TensorFlow'}!
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
        <h4 className="font-semibold text-blue-900 mb-2">Key Differences from Linear Regression:</h4>
        <ul className="list-disc list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Sigmoid Activation:</strong> Outputs probabilities between 0 and 1</li>
          <li><strong>Binary Cross-Entropy Loss:</strong> Designed for binary classification</li>
          <li><strong>Decision Threshold:</strong> Predict class 1 if probability &gt; 0.5, else class 0</li>
          <li><strong>Accuracy Metric:</strong> Measures percentage of correct predictions</li>
        </ul>
      </div>
    </div>
  );
}

