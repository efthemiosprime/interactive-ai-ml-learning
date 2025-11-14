import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function ModelEvaluation({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Simple binary classifier
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)

# Generate test data
x_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100, 1)).float()

# Evaluate model
model.eval()
with torch.no_grad():
    predictions = model(x_test)
    predicted_classes = (predictions > 0.5).float().numpy()
    y_true = y_test.numpy()

# Calculate metrics
accuracy = accuracy_score(y_true, predicted_classes)
precision = precision_score(y_true, predicted_classes)
recall = recall_score(y_true, predicted_classes)
f1 = f1_score(y_true, predicted_classes)

print("Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.2%}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Simple binary classifier
model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(10,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Generate test data
x_test = np.random.randn(100, 10).astype(np.float32)
y_test = np.random.randint(0, 2, (100, 1)).astype(np.float32)

# Evaluate model
predictions = model.predict(x_test, verbose=0)
predicted_classes = (predictions > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes)

print("Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.2%}")`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = `Model Evaluation Metrics:
Accuracy:  78.00%
Precision: 75.51%
Recall:    82.35%
F1-Score:  78.78%`;

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
          💡 <strong>Model Evaluation:</strong> Measure model performance using various metrics!
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
        <h4 className="font-semibold text-blue-900 mb-2">Evaluation Metrics:</h4>
        <ul className="list-disc list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Accuracy:</strong> Percentage of correct predictions</li>
          <li><strong>Precision:</strong> Of predicted positives, how many are actually positive?</li>
          <li><strong>Recall:</strong> Of actual positives, how many did we find?</li>
          <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
        </ul>
      </div>
    </div>
  );
}

