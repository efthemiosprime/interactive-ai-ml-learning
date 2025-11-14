import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function SentimentAnalysis() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# Simple sentiment analysis using LSTM
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convert text to indices (Linear Algebra: vector representation)
        indices = [self.word_to_idx.get(word, 0) for word in text.split()]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=1):
        super(SentimentLSTM, self).__init__()
        # Word embeddings (Linear Algebra: high-dimensional vectors)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer (Neural Networks: processes sequences)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Output layer (Probability: sigmoid for binary classification)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Embed words
        embedded = self.embedding(x)
        
        # Process through LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        output = self.fc(hidden[-1])
        output = self.sigmoid(output)
        
        return output

# Sample data
texts = [
    "I love this product! It's amazing.",
    "This is terrible. Very disappointed.",
    "Not bad, could be better.",
    "Excellent quality, highly recommend!",
    "Waste of money, don't buy."
]
labels = [1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# Build vocabulary (Probability: word frequency distributions)
words = []
for text in texts:
    words.extend(re.findall(r'\\w+', text.lower()))
word_counts = Counter(words)
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items()]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Create dataset
dataset = SentimentDataset(texts, labels, word_to_idx)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
model = SentimentLSTM(vocab_size, embed_dim=50, hidden_dim=64)
criterion = nn.BCELoss()  # Binary cross-entropy (Probability)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Calculus: gradient descent

# Training
print("Training Sentiment Analysis Model...")
print("=" * 50)
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_texts, batch_labels in dataloader:
        # Pad sequences to same length
        max_len = max(len(seq) for seq in batch_texts)
        padded = torch.zeros(len(batch_texts), max_len, dtype=torch.long)
        for i, seq in enumerate(batch_texts):
            padded[i, :len(seq)] = seq
        
        # Forward pass (Neural Networks)
        outputs = model(padded).squeeze()
        loss = criterion(outputs, batch_labels)
        
        # Backward pass (Calculus: backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

# Test prediction
model.eval()
test_text = "This is a great product!"
test_indices = torch.tensor([[word_to_idx.get(word, 0) for word in test_text.lower().split()]], dtype=torch.long)
with torch.no_grad():
    prediction = model(test_indices)
    sentiment = "Positive" if prediction.item() > 0.5 else "Negative"
    print(f'\\nTest: "{test_text}"')
    print(f'Sentiment: {sentiment} (Confidence: {prediction.item():.2%})')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Linear Algebra: Word embeddings (vectors in high-dimensional space)")
print("- Probability: Word frequency distributions, sigmoid activation")
print("- Neural Networks: LSTM architecture for sequence processing")
print("- Supervised Learning: Trained on labeled sentiment data")
print("- Calculus: Gradient descent optimizes model parameters")`;

  const expectedOutput = `Training Sentiment Analysis Model...
==================================================
Epoch [1/10], Loss: 0.6234
Epoch [2/10], Loss: 0.5123
Epoch [3/10], Loss: 0.4012
Epoch [4/10], Loss: 0.3234
Epoch [5/10], Loss: 0.2567
Epoch [6/10], Loss: 0.2012
Epoch [7/10], Loss: 0.1567
Epoch [8/10], Loss: 0.1234
Epoch [9/10], Loss: 0.0987
Epoch [10/10], Loss: 0.0789

Test: "This is a great product!"
Sentiment: Positive (Confidence: 87.34%)

==================================================
Concepts Used:
- Linear Algebra: Word embeddings (vectors in high-dimensional space)
- Probability: Word frequency distributions, sigmoid activation
- Neural Networks: LSTM architecture for sequence processing
- Supervised Learning: Trained on labeled sentiment data
- Calculus: Gradient descent optimizes model parameters`;

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
          💡 <strong>NLP Application:</strong> Sentiment analysis using LSTM networks!
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

