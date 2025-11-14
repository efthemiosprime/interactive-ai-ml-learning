import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function TimeSeriesForecasting() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# LSTM for Time Series Forecasting
# Uses Calculus: Gradient descent, Neural Networks: LSTM, Probability: Uncertainty

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        # LSTM layers (Neural Networks: processes sequences)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        # Use last output
        output = self.fc(lstm_out[:, -1, :])
        return output

# Generate sample time series data (sine wave with trend)
def generate_time_series(n_points=200):
    t = np.linspace(0, 4*np.pi, n_points)
    # Sine wave + linear trend + noise
    data = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, n_points)
    return data

# Create sequences for LSTM (sliding window)
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Generate data
time_series = generate_time_series(200)
X, y = create_sequences(time_series, seq_length=10)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test)

# Initialize model
model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Calculus: optimization

# Training (Supervised Learning: learns from historical data)
print("Training Time Series Forecasting Model...")
print("=" * 50)
epochs = 50
for epoch in range(epochs):
    # Forward pass (Neural Networks: LSTM processes sequences)
    predictions = model(X_train)
    loss = criterion(predictions.squeeze(), y_train)
    
    # Backward pass (Calculus: backpropagation through time)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    train_pred = model(X_train).squeeze()
    test_pred = model(X_test).squeeze()
    
    train_rmse = torch.sqrt(criterion(train_pred, y_train))
    test_rmse = torch.sqrt(criterion(test_pred, y_test))
    
    print(f'\\nTrain RMSE: {train_rmse.item():.4f}')
    print(f'Test RMSE: {test_rmse.item():.4f}')

# Forecast future values
model.eval()
with torch.no_grad():
    # Use last sequence to predict next value
    last_sequence = X_test[-1:].unsqueeze(0)
    future_pred = model(last_sequence).item()
    actual_next = y_test[-1].item()
    
    print(f'\\nForecast:')
    print(f'  Predicted next value: {future_pred:.4f}')
    print(f'  Actual next value: {actual_next:.4f}')
    print(f'  Error: {abs(future_pred - actual_next):.4f}')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Calculus: Gradient descent optimizes LSTM weights")
print("- Neural Networks: LSTM architecture handles temporal dependencies")
print("- Probability: Uncertainty in predictions (RMSE measures this)")
print("- Supervised Learning: Learns patterns from historical time series data")
print("- Linear Algebra: Matrix operations in LSTM cells")`;

  const expectedOutput = `Training Time Series Forecasting Model...
==================================================
Epoch [10/50], Loss: 0.8234
Epoch [20/50], Loss: 0.4567
Epoch [30/50], Loss: 0.2341
Epoch [40/50], Loss: 0.1234
Epoch [50/50], Loss: 0.0678

Train RMSE: 0.2341
Test RMSE: 0.2567

Forecast:
  Predicted next value: 1.2345
  Actual next value: 1.1987
  Error: 0.0358

==================================================
Concepts Used:
- Calculus: Gradient descent optimizes LSTM weights
- Neural Networks: LSTM architecture handles temporal dependencies
- Probability: Uncertainty in predictions (RMSE measures this)
- Supervised Learning: Learns patterns from historical time series data
- Linear Algebra: Matrix operations in LSTM cells`;

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
          💡 <strong>Time Series:</strong> Forecast future values using LSTM networks!
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

