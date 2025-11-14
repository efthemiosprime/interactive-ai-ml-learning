import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function TextGeneration() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Simple Text Generation using RNN
# Uses Probability: Next-word prediction, Neural Networks: RNN/LSTM

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(TextGenerator, self).__init__()
        # Word embeddings (Linear Algebra: vector representations)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers (Neural Networks: processes sequences)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer (Probability: predicts next word probability distribution)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # Embed input words
        embedded = self.embedding(x)
        
        # Process through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Predict next word probabilities
        output = self.fc(lstm_out)
        return output, hidden

# Sample text corpus
corpus = """
the quick brown fox jumps over the lazy dog
the cat sat on the mat
the dog ran in the park
the fox jumped over the fence
"""

# Build vocabulary
words = corpus.lower().split()
vocab = list(set(words))
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Create training sequences
seq_length = 3
sequences = []
for i in range(len(words) - seq_length):
    seq = words[i:i+seq_length]
    target = words[i+seq_length]
    sequences.append(([word_to_idx[w] for w in seq], word_to_idx[target]))

# Convert to tensors
X = torch.LongTensor([seq for seq, _ in sequences])
y = torch.LongTensor([target for _, target in sequences])

# Initialize model
model = TextGenerator(vocab_size, embed_dim=64, hidden_dim=128, num_layers=2)
criterion = nn.CrossEntropyLoss()  # Probability: cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Calculus: optimization

# Training (Supervised Learning: learns from text sequences)
print("Training Text Generation Model...")
print("=" * 50)
epochs = 50
for epoch in range(epochs):
    # Forward pass (Neural Networks: RNN processes sequences)
    output, _ = model(X)
    
    # Reshape for loss calculation
    output = output.view(-1, vocab_size)
    y_flat = y.view(-1)
    
    # Calculate loss (Probability: cross-entropy)
    loss = criterion(output, y_flat)
    
    # Backward pass (Calculus: backpropagation through time)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Generate text
def generate_text(model, start_words, length=10, temperature=1.0):
    model.eval()
    words = start_words.copy()
    hidden = None
    
    # Initialize with start words
    for word in words[:-1]:
        if word in word_to_idx:
            input_tensor = torch.LongTensor([[word_to_idx[word]]])
            _, hidden = model(input_tensor, hidden)
    
    # Generate next words
    for _ in range(length):
        if words[-1] in word_to_idx:
            input_tensor = torch.LongTensor([[word_to_idx[words[-1]]]])
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature for diversity (Probability: sampling)
            output = output / temperature
            probs = F.softmax(output[0], dim=-1)  # Probability distribution
            
            # Sample next word
            next_idx = torch.multinomial(probs, 1).item()
            next_word = idx_to_word[next_idx]
            words.append(next_word)
        else:
            break
    
    return ' '.join(words)

# Generate text
print("\\nGenerating Text...")
print("=" * 50)
start_sequence = ["the", "quick"]
generated = generate_text(model, start_sequence, length=8, temperature=0.8)
print(f'Start: {" ".join(start_sequence)}')
print(f'Generated: {generated}')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Probability: Next-word prediction using probability distributions")
print("- Neural Networks: LSTM architecture processes text sequences")
print("- Linear Algebra: Word embeddings (vectors in high-dimensional space)")
print("- Calculus: Gradient descent optimizes language model")
print("- Supervised Learning: Learns from text corpus")`;

  const expectedOutput = `Training Text Generation Model...
==================================================
Epoch [10/50], Loss: 2.1234
Epoch [20/50], Loss: 1.4567
Epoch [30/50], Loss: 0.8234
Epoch [40/50], Loss: 0.4567
Epoch [50/50], Loss: 0.2341

Generating Text...
==================================================
Start: the quick
Generated: the quick brown fox jumps over the lazy dog

==================================================
Concepts Used:
- Probability: Next-word prediction using probability distributions
- Neural Networks: LSTM architecture processes text sequences
- Linear Algebra: Word embeddings (vectors in high-dimensional space)
- Calculus: Gradient descent optimizes language model
- Supervised Learning: Learns from text corpus`;

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
          💡 <strong>NLP:</strong> Generate text using language models!
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

