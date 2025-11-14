import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function DataLoading({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Generate sample data
x = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)

# Create dataset and dataloader
dataset = CustomDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
print("Dataset size:", len(dataset))
print("\\nFirst batch:")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  X shape: {batch_x.shape}")
    print(f"  Y shape: {batch_y.shape}")
    if batch_idx == 0:
        break`;

  const tensorflowCode = `import tensorflow as tf
import numpy as np

# Generate sample data
x = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randn(1000, 1).astype(np.float32)

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(1000).batch(32)

# Iterate through batches
print("Dataset size: 1000")
print("\\nFirst batch:")
for batch_idx, (batch_x, batch_y) in enumerate(dataset):
    print(f"Batch {batch_idx + 1}:")
    print(f"  X shape: {batch_x.shape}")
    print(f"  Y shape: {batch_y.shape}")
    if batch_idx == 0:
        break`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = `Dataset size: 1000

First batch:
Batch 1:
  X shape: torch.Size([32, 10])
  Y shape: torch.Size([32, 1])`;

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
          💡 <strong>Data Loading:</strong> Learn how to load and batch data efficiently!
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
        <h4 className="font-semibold text-blue-900 mb-2">Key Concepts:</h4>
        <ul className="list-disc list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Batching:</strong> Process multiple samples together for efficiency</li>
          <li><strong>Shuffling:</strong> Randomize data order to improve training</li>
          <li><strong>Iteration:</strong> Loop through batches during training</li>
          <li><strong>Custom Datasets:</strong> Create datasets for your specific data format</li>
        </ul>
      </div>
    </div>
  );
}

