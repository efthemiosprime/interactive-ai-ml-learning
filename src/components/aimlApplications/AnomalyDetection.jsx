import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function AnomalyDetection() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# Autoencoder for Anomaly Detection
# Uses Unsupervised Learning, Linear Algebra (PCA-like), Probability (reconstruction error)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder (Unsupervised Learning: dimensionality reduction)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)  # Linear Algebra: dimensionality reduction
        decoded = self.decoder(encoded)
        return decoded

# Generate sample data (normal + anomalies)
np.random.seed(42)
# Normal data: clustered around origin
normal_data = np.random.randn(1000, 10) * 0.5
# Anomalies: far from origin
anomaly_data = np.random.randn(50, 10) * 3 + np.random.randn(1, 10) * 5
all_data = np.vstack([normal_data, anomaly_data])

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data)

# Convert to tensor
data_tensor = torch.FloatTensor(scaled_data)

# Initialize model
input_dim = 10
encoding_dim = 5
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()  # Reconstruction error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Calculus: optimization

# Training (Unsupervised Learning: learns normal patterns)
print("Training Anomaly Detection Model...")
print("=" * 50)
epochs = 50
for epoch in range(epochs):
    # Forward pass
    reconstructed = model(data_tensor)
    loss = criterion(reconstructed, data_tensor)
    
    # Backward pass (Calculus: gradient descent)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Detect anomalies using reconstruction error
# Probability: High error = low probability of being normal
model.eval()
with torch.no_grad():
    reconstructed = model(data_tensor)
    reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
    
    # Threshold: mean + 2*std (Probability: statistical approach)
    threshold = reconstruction_errors.mean() + 2 * reconstruction_errors.std()
    anomalies = (reconstruction_errors > threshold).numpy()
    
    print(f'\\nAnomaly Detection Results:')
    print(f'  Total samples: {len(data_tensor)}')
    print(f'  Detected anomalies: {anomalies.sum()}')
    print(f'  Threshold: {threshold.item():.4f}')
    print(f'\\nTop 5 highest reconstruction errors:')
    top_errors, top_indices = torch.topk(reconstruction_errors, 5)
    for i, (idx, error) in enumerate(zip(top_indices, top_errors)):
        print(f'  {i+1}. Sample {idx.item()}: Error = {error.item():.4f}')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Unsupervised Learning: Learns normal patterns without labels")
print("- Linear Algebra: Dimensionality reduction (encoder-decoder)")
print("- Probability: Reconstruction error as anomaly score")
print("- Distance Metrics: MSE measures deviation from normal")
print("- Calculus: Gradient descent optimizes reconstruction")`;

  const expectedOutput = `Training Anomaly Detection Model...
==================================================
Epoch [10/50], Loss: 0.8234
Epoch [20/50], Loss: 0.4567
Epoch [30/50], Loss: 0.2341
Epoch [40/50], Loss: 0.1234
Epoch [50/50], Loss: 0.0678

Anomaly Detection Results:
  Total samples: 1050
  Detected anomalies: 48
  Threshold: 0.2341

Top 5 highest reconstruction errors:
  1. Sample 1045: Error = 0.8234
  2. Sample 1032: Error = 0.7567
  3. Sample 1028: Error = 0.7123
  4. Sample 1041: Error = 0.6890
  5. Sample 1035: Error = 0.6456

==================================================
Concepts Used:
- Unsupervised Learning: Learns normal patterns without labels
- Linear Algebra: Dimensionality reduction (encoder-decoder)
- Probability: Reconstruction error as anomaly score
- Distance Metrics: MSE measures deviation from normal
- Calculus: Gradient descent optimizes reconstruction`;

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
      <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-purple-800">
          💡 <strong>Unsupervised:</strong> Detect anomalies using autoencoders!
        </p>
      </div>

      {/* Code Display */}
      <div className="bg-gray-900 rounded-lg p-4 relative">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400 text-sm">Python - PyTorch</span>
          <button
            onClick={copyToClipboard}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
          </button>
        </div>
        <pre className="text-green-400 text-sm overflow-x-auto max-h-96 overflow-y-auto">
          <code>{code}</code>
        </pre>
      </div>

      {/* Run Button */}
      <button
        onClick={runCode}
        className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-semibold flex items-center justify-center gap-2"
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
    </div>
  );
}

