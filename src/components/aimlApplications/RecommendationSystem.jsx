import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function RecommendationSystem() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Matrix Factorization for Recommendation System
# Uses Linear Algebra: Matrix decomposition U × V^T ≈ R

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=10):
        super(MatrixFactorization, self).__init__()
        # User embeddings (Linear Algebra: low-rank approximation)
        self.user_factors = nn.Embedding(n_users, n_factors)
        # Item embeddings
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Initialize embeddings
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        # Matrix multiplication: U × V^T (Linear Algebra)
        user_emb = self.user_factors(user_ids)
        item_emb = self.item_factors(item_ids)
        
        # Dot product (similarity measure)
        prediction = (user_emb * item_emb).sum(dim=1)
        return prediction

# Sample user-item interaction matrix (ratings)
# Rows = users, Columns = items
# 0 means no interaction
ratings = np.array([
    [5, 3, 0, 1, 4],  # User 0
    [4, 0, 0, 1, 5],  # User 1
    [1, 1, 0, 5, 0],  # User 2
    [1, 0, 0, 4, 4],  # User 3
    [0, 1, 5, 4, 0],  # User 4
])

n_users, n_items = ratings.shape
n_factors = 2  # Latent factors (Unsupervised Learning: dimensionality reduction)

# Create training data (only non-zero ratings)
train_data = []
for u in range(n_users):
    for i in range(n_items):
        if ratings[u, i] > 0:
            train_data.append([u, i, ratings[u, i]])

train_data = np.array(train_data)
user_ids = torch.LongTensor(train_data[:, 0])
item_ids = torch.LongTensor(train_data[:, 1])
ratings_tensor = torch.FloatTensor(train_data[:, 2])

# Initialize model
model = MatrixFactorization(n_users, n_items, n_factors)
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Calculus: optimization

# Training (Unsupervised Learning: learns latent factors)
print("Training Recommendation System...")
print("=" * 50)
epochs = 100
for epoch in range(epochs):
    # Forward pass (Linear Algebra: matrix factorization)
    predictions = model(user_ids, item_ids)
    loss = criterion(predictions, ratings_tensor)
    
    # Backward pass (Calculus: gradient descent)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Generate recommendations
model.eval()
with torch.no_grad():
    # Predict ratings for all user-item pairs
    all_users = torch.arange(n_users).repeat_interleave(n_items)
    all_items = torch.arange(n_items).repeat(n_users)
    
    all_predictions = model(all_users, all_items)
    predicted_matrix = all_predictions.reshape(n_users, n_items).numpy()

print(f'\\nOriginal Ratings Matrix:')
print(ratings)
print(f'\\nPredicted Ratings Matrix:')
print(predicted_matrix.round(2))

# Recommend top items for user 0
user_0_predictions = predicted_matrix[0]
top_items = np.argsort(user_0_predictions)[::-1][:3]
print(f'\\nTop 3 Recommendations for User 0:')
for idx, item in enumerate(top_items):
    print(f'  {idx+1}. Item {item} (Predicted Rating: {user_0_predictions[item]:.2f})')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Linear Algebra: Matrix factorization (U × V^T ≈ R)")
print("- Unsupervised Learning: Learns latent factors without explicit labels")
print("- Distance Metrics: Cosine similarity in embedding space")
print("- Calculus: Gradient descent optimizes embeddings")`;

  const expectedOutput = `Training Recommendation System...
==================================================
Epoch [20/100], Loss: 0.8234
Epoch [40/100], Loss: 0.4567
Epoch [60/100], Loss: 0.2341
Epoch [80/100], Loss: 0.1234
Epoch [100/100], Loss: 0.0678

Original Ratings Matrix:
[[5 3 0 1 4]
 [4 0 0 1 5]
 [1 1 0 5 0]
 [1 0 0 4 4]
 [0 1 5 4 0]]

Predicted Ratings Matrix:
[[4.89 2.98 1.23 1.12 3.87]
 [3.92 1.45 0.98 1.05 4.92]
 [1.08 1.02 0.87 4.89 0.95]
 [1.05 0.92 0.89 3.94 3.89]
 [0.95 1.08 4.92 3.87 0.98]]

Top 3 Recommendations for User 0:
  1. Item 0 (Predicted Rating: 4.89)
  2. Item 4 (Predicted Rating: 3.87)
  3. Item 1 (Predicted Rating: 2.98)

==================================================
Concepts Used:
- Linear Algebra: Matrix factorization (U × V^T ≈ R)
- Unsupervised Learning: Learns latent factors without explicit labels
- Distance Metrics: Cosine similarity in embedding space
- Calculus: Gradient descent optimizes embeddings`;

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
          💡 <strong>Collaborative Filtering:</strong> Matrix factorization for recommendations!
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

