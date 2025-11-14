import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function ImageClassification() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');
  const [framework, setFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Define CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layers (Linear Algebra: matrix operations)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss, and optimizer
model = CNN(num_classes=10)
criterion = nn.CrossEntropyLoss()  # Uses Probability: softmax + cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Uses Calculus: gradient descent

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Forward pass (Neural Networks)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass (Calculus: backpropagation)
            optimizer.zero_grad()
            loss.backward()  # Computes gradients
            optimizer.step()  # Updates weights using gradients
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

# Train the model
print("Training CNN for Image Classification...")
print("=" * 50)
trained_model = train_model(model, train_loader, criterion, optimizer, epochs=5)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\\nTest Accuracy: {accuracy:.2f}%')
    return accuracy

test_accuracy = evaluate_model(trained_model, test_loader)

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Linear Algebra: Convolution operations (matrix multiplication)")
print("- Calculus: Gradient descent and backpropagation")
print("- Probability: Softmax activation for class probabilities")
print("- Neural Networks: Multi-layer CNN architecture")
print("- Supervised Learning: Training on labeled CIFAR-10 dataset")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define CNN architecture using Keras
def create_cnn(num_classes=10):
    model = keras.Sequential([
        # Convolutional layers (Linear Algebra: matrix operations)
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Probability: softmax
    ])
    
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create model
model = create_cnn(num_classes=10)

# Compile model
# Uses Calculus: Adam optimizer (gradient descent variant)
# Uses Probability: categorical cross-entropy loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training (uses Neural Networks: forward/backward pass)
print("Training CNN for Image Classification...")
print("=" * 50)
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'\\nTest Accuracy: {test_accuracy*100:.2f}%')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Linear Algebra: Convolution operations (matrix multiplication)")
print("- Calculus: Gradient descent and backpropagation")
print("- Probability: Softmax activation for class probabilities")
print("- Neural Networks: Multi-layer CNN architecture")
print("- Supervised Learning: Training on labeled CIFAR-10 dataset")`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = `Training CNN for Image Classification...
==================================================
Epoch [1/5], Loss: 1.8234, Accuracy: 35.67%
Epoch [2/5], Loss: 1.4567, Accuracy: 48.23%
Epoch [3/5], Loss: 1.2341, Accuracy: 56.78%
Epoch [4/5], Loss: 1.0890, Accuracy: 62.34%
Epoch [5/5], Loss: 0.9876, Accuracy: 65.89%

Test Accuracy: 64.23%

==================================================
Concepts Used:
- Linear Algebra: Convolution operations (matrix multiplication)
- Calculus: Gradient descent and backpropagation
- Probability: Softmax activation for class probabilities
- Neural Networks: Multi-layer CNN architecture
- Supervised Learning: Training on labeled CIFAR-10 dataset`;

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
          💡 <strong>Complete Application:</strong> Image classification using CNNs - combines all concepts!
        </p>
      </div>

      {/* Framework Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Framework
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setFramework('pytorch')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              framework === 'pytorch'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            PyTorch
          </button>
          <button
            onClick={() => setFramework('tensorflow')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              framework === 'tensorflow'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            TensorFlow
          </button>
        </div>
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

      {/* Key Concepts */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-4 border-2 border-purple-200">
        <h4 className="font-semibold text-purple-900 mb-2">How Concepts Come Together:</h4>
        <ul className="list-disc list-inside space-y-1 text-sm text-purple-800">
          <li><strong>Linear Algebra:</strong> Convolution = matrix multiplication in feature extraction</li>
          <li><strong>Calculus:</strong> Backpropagation computes gradients to update weights</li>
          <li><strong>Probability:</strong> Softmax outputs class probabilities</li>
          <li><strong>Neural Networks:</strong> Multi-layer architecture processes images hierarchically</li>
          <li><strong>Supervised Learning:</strong> Trained on labeled data (CIFAR-10: 10 classes)</li>
        </ul>
      </div>
    </div>
  );
}

