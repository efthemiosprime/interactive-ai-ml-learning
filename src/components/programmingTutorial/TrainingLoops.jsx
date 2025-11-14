import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function TrainingLoops({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Generate data
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataloader = DataLoader(list(zip(x, y)), batch_size=32, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
import numpy as np

# Simple model
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(10,), activation='relu'),
    keras.layers.Dense(1)
])

# Generate data
x = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randn(1000, 1).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(1000).batch(32)

# Compile model
model.compile(optimizer='adam', loss='mse')

# Training loop (manual)
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = tf.reduce_mean(tf.square(predictions - batch_y))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += loss.numpy()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = `Epoch [1/10], Average Loss: 0.8234
Epoch [2/10], Average Loss: 0.6543
Epoch [3/10], Average Loss: 0.5123
Epoch [4/10], Average Loss: 0.4012
Epoch [5/10], Average Loss: 0.3145
Epoch [6/10], Average Loss: 0.2467
Epoch [7/10], Average Loss: 0.1934
Epoch [8/10], Average Loss: 0.1512
Epoch [9/10], Average Loss: 0.1187
Epoch [10/10], Average Loss: 0.0934`;

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
          💡 <strong>Complete Training Loop:</strong> Implement forward pass, loss calculation, backpropagation, and optimization!
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
        <h4 className="font-semibold text-blue-900 mb-2">Training Loop Steps:</h4>
        <ol className="list-decimal list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Forward Pass:</strong> Compute predictions from input</li>
          <li><strong>Loss Calculation:</strong> Compare predictions with true values</li>
          <li><strong>Backward Pass:</strong> Compute gradients using automatic differentiation</li>
          <li><strong>Optimizer Step:</strong> Update model weights using gradients</li>
          <li><strong>Repeat:</strong> Iterate over epochs and batches</li>
        </ol>
      </div>
    </div>
  );
}

