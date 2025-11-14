import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function NBAChatbotTutorial() {
  const [selectedStep, setSelectedStep] = useState('step1');
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const steps = {
    step1: {
      title: 'Step 1: Data Collection & Preprocessing',
      description: 'Gather NBA data and prepare it for training',
      code: `# Step 1: Data Collection & Preprocessing
import pandas as pd
import numpy as np
import json
from collections import Counter

# 1.1 Collect NBA Data
# Sample NBA Q&A pairs
nba_data = [
    {"question": "Who won the NBA championship in 2023?", "answer": "The Denver Nuggets won the NBA championship in 2023."},
    {"question": "What team does LeBron James play for?", "answer": "LeBron James plays for the Los Angeles Lakers."},
    {"question": "Who is the all-time leading scorer in NBA history?", "answer": "LeBron James is the all-time leading scorer in NBA history."},
    {"question": "How many teams are in the NBA?", "answer": "There are 30 teams in the NBA."},
    {"question": "What is the NBA Finals MVP?", "answer": "The NBA Finals MVP is awarded to the best player in the championship series."},
    # ... more Q&A pairs
]

# 1.2 Text Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Preprocess all questions and answers
processed_data = []
for item in nba_data:
    processed_data.append({
        'question': preprocess_text(item['question']),
        'answer': preprocess_text(item['answer'])
    })

# 1.3 Build Vocabulary
# Linear Algebra: Words as vectors in high-dimensional space
def build_vocabulary(data, min_freq=2):
    """Build vocabulary from processed data"""
    word_counts = Counter()
    
    for item in data:
        for word in item['question'].split() + item['answer'].split():
            word_counts[word] += 1
    
    # Filter by minimum frequency
    vocabulary = {word: idx + 2 for idx, (word, count) in enumerate(
        [(w, c) for w, c in word_counts.items() if c >= min_freq]
    )}
    
    # Add special tokens
    vocabulary['<PAD>'] = 0
    vocabulary['<UNK>'] = 1
    
    return vocabulary

vocab = build_vocabulary(processed_data)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# 1.4 Convert Text to Sequences
# Linear Algebra: Text → Vector representation
def text_to_sequence(text, vocab, max_length=50):
    """Convert text to sequence of integers"""
    tokens = text.split()
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad or truncate to max_length
    if len(sequence) < max_length:
        sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    
    return sequence

# Prepare training data
max_seq_length = 50
X = [text_to_sequence(item['question'], vocab, max_seq_length) for item in processed_data]
y = [text_to_sequence(item['answer'], vocab, max_seq_length) for item in processed_data]

X = np.array(X)
y = np.array(y)

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")`
    },
    step2: {
      title: 'Step 2: Mathematical Foundations - Embeddings',
      description: 'Understand word embeddings using Linear Algebra',
      code: `# Step 2: Mathematical Foundations - Word Embeddings
import torch
import torch.nn as nn
import numpy as np

# 2.1 Word Embeddings (Linear Algebra)
# Each word is represented as a dense vector in high-dimensional space
# Similar words are close together in this space

class WordEmbeddings(nn.Module):
    """
    Word Embedding Layer
    
    Mathematical Representation:
    - Input: word index i (integer)
    - Output: embedding vector e_i ∈ R^d
    
    Formula: e_i = E[i] where E ∈ R^(V×d)
    - V = vocabulary size
    - d = embedding dimension
    """
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        # Linear Algebra: Embedding matrix E
        # Shape: [vocab_size, embedding_dim]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings (Xavier uniform)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x):
        """
        x: [batch_size, seq_length] - word indices
        Returns: [batch_size, seq_length, embedding_dim] - embedded vectors
        """
        return self.embedding(x)

# Example: Embedding a sentence
vocab_size = 10000
embedding_dim = 128
embedding_layer = WordEmbeddings(vocab_size, embedding_dim)

# Input: sentence as word indices [batch=1, seq_length=5]
sentence = torch.tensor([[1, 5, 3, 8, 2]])

# Output: embedded sentence [batch=1, seq_length=5, embedding_dim=128]
embedded = embedding_layer(sentence)
print(f"Embedded shape: {embedded.shape}")

# 2.2 Positional Encoding (Linear Algebra + Trigonometry)
# Add position information to embeddings
class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    
    Formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
    - pos = position in sequence
    - i = dimension index
    - d_model = embedding dimension
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch_size, seq_length, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

# 2.3 Cosine Similarity (Linear Algebra)
# Measure similarity between word embeddings
def cosine_similarity(vec1, vec2):
    """
    Cosine Similarity Formula:
    similarity = (vec1 · vec2) / (||vec1|| × ||vec2||)
    
    Returns value between -1 and 1
    """
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-8)

# Example: Find similar words
word1_embedding = embedding_layer(torch.tensor([[1]]))[0, 0]
word2_embedding = embedding_layer(torch.tensor([[5]]))[0, 0]
similarity = cosine_similarity(word1_embedding, word2_embedding)
print(f"Cosine similarity: {similarity.item():.4f}")`
    },
    step3: {
      title: 'Step 3: Neural Network Architecture - LSTM/Transformer',
      description: 'Build the chatbot model using Neural Networks',
      code: `# Step 3: Neural Network Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 3.1 LSTM-based Seq2Seq Model
class LSTMSeq2Seq(nn.Module):
    """
    Sequence-to-Sequence Model with LSTM
    
    Architecture:
    - Encoder LSTM: Processes input question
    - Decoder LSTM: Generates answer
    
    Mathematical Formulation:
    - Hidden state: h_t = LSTM(x_t, h_{t-1})
    - Output: y_t = softmax(W_o · h_t + b_o)
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, encoder_input, decoder_input=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        
        Formula for LSTM:
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
        C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
        C_t = f_t * C_{t-1} + i_t * C̃_t  # Cell state
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
        h_t = o_t * tanh(C_t)  # Hidden state
        """
        batch_size = encoder_input.size(0)
        
        # Encoder
        encoder_embedded = self.embedding(encoder_input)
        encoder_output, (hidden, cell) = self.encoder(encoder_embedded)
        
        # Decoder
        if decoder_input is not None:
            decoder_embedded = self.embedding(decoder_input)
            decoder_output, _ = self.decoder(decoder_embedded, (hidden, cell))
            output = self.output_proj(decoder_output)
            return output
        
        # Inference: Generate token by token
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long).to(encoder_input.device)
        decoder_hidden = hidden
        decoder_cell = cell
        
        for _ in range(50):  # Max length
            decoder_embedded = self.embedding(decoder_input)
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_embedded, (decoder_hidden, decoder_cell)
            )
            output = self.output_proj(decoder_output)
            outputs.append(output)
            
            # Get next token (greedy)
            next_token = output.argmax(dim=-1)
            decoder_input = next_token
        
        return torch.cat(outputs, dim=1)

# 3.2 Transformer-based Model (More Modern)
class TransformerChatbot(nn.Module):
    """
    Transformer-based Chatbot
    
    Uses Multi-Head Attention mechanism
    
    Attention Formula:
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    
    where:
    - Q = Query matrix
    - K = Key matrix  
    - V = Value matrix
    - d_k = dimension of keys
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, src, src_mask=None):
        """
        Forward pass through Transformer
        """
        # Embedding + Positional Encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer(src, src_key_padding_mask=src_mask)
        
        # Output projection
        output = self.output_proj(output)
        
        return output

# Initialize model
vocab_size = 10000
model = LSTMSeq2Seq(vocab_size, embedding_dim=128, hidden_dim=256)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")`
    },
    step4: {
      title: 'Step 4: Training - Loss Function & Optimization',
      description: 'Train the model using Calculus (Gradient Descent)',
      code: `# Step 4: Training - Loss Function & Optimization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 4.1 Loss Function (Cross-Entropy)
"""
Cross-Entropy Loss Formula:

L = -Σ y_i · log(ŷ_i)

where:
- y_i = true probability distribution (one-hot)
- ŷ_i = predicted probability distribution
- Sum over all classes

For sequence-to-sequence:
L = -Σ_t Σ_i y_{t,i} · log(ŷ_{t,i})

where t = time step, i = vocabulary index
"""

class ChatbotDataset(Dataset):
    def __init__(self, questions, answers, vocab, max_length=50):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.answers[idx], dtype=torch.long)
        )

# 4.2 Training Function
def train_chatbot(model, train_loader, epochs=10, learning_rate=0.001):
    """
    Training Loop
    
    Uses Gradient Descent:
    θ_{t+1} = θ_t - α · ∇_θ L(θ_t)
    
    where:
    - θ = model parameters
    - α = learning rate
    - ∇_θ L = gradient of loss function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function (Cross-Entropy)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Optimizer (Adam - Adaptive Moment Estimation)
    """
    Adam Optimizer Formula:
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t  # First moment
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²  # Second moment
    m̂_t = m_t / (1 - β₁^t)  # Bias correction
    v̂_t = v_t / (1 - β₂^t)
    θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (questions, answers) in enumerate(train_loader):
            questions = questions.to(device)
            answers = answers.to(device)
            
            # Forward pass
            # Input: questions, Output: predicted answer tokens
            decoder_input = answers[:, :-1]  # Remove last token
            decoder_target = answers[:, 1:]   # Remove first token (shifted)
            
            predictions = model(questions, decoder_input)
            
            # Reshape for loss calculation
            # [batch_size, seq_length, vocab_size] -> [batch_size * seq_length, vocab_size]
            predictions = predictions.reshape(-1, predictions.size(-1))
            decoder_target = decoder_target.reshape(-1)
            
            # Calculate loss
            loss = criterion(predictions, decoder_target)
            
            # Backward pass (Calculus: Chain Rule)
            """
            Backpropagation uses Chain Rule:
            ∂L/∂θ = ∂L/∂ŷ · ∂ŷ/∂h · ∂h/∂θ
            
            where:
            - L = loss
            - ŷ = prediction
            - h = hidden state
            - θ = parameters
            """
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        scheduler.step()
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    return model

# 4.3 Example Training
# Prepare data
train_dataset = ChatbotDataset(X, y, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = LSTMSeq2Seq(vocab_size=len(vocab), embedding_dim=128, hidden_dim=256)

# Train
trained_model = train_chatbot(model, train_loader, epochs=10)`
    },
    step5: {
      title: 'Step 5: Probability & Inference',
      description: 'Generate responses using Probability distributions',
      code: `# Step 5: Probability & Inference
import torch
import torch.nn.functional as F
import numpy as np

# 5.1 Greedy Decoding
def greedy_decode(model, question, vocab, max_length=50):
    """
    Greedy Decoding: Always pick most likely token
    
    Formula:
    y_t = argmax P(y_t | y_{<t}, x)
    
    where:
    - y_t = token at time t
    - y_{<t} = previous tokens
    - x = input question
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Convert question to tensor
    question_tensor = torch.tensor([question], dtype=torch.long).to(device)
    
    # Encode question
    with torch.no_grad():
        # Get encoder output
        encoder_embedded = model.embedding(question_tensor)
        encoder_output, (hidden, cell) = model.encoder(encoder_embedded)
        
        # Decode token by token
        decoded_tokens = []
        decoder_input = torch.zeros(1, 1, dtype=torch.long).to(device)
        decoder_hidden = hidden
        decoder_cell = cell
        
        for _ in range(max_length):
            decoder_embedded = model.embedding(decoder_input)
            decoder_output, (decoder_hidden, decoder_cell) = model.decoder(
                decoder_embedded, (decoder_hidden, decoder_cell)
            )
            
            # Get probability distribution
            output = model.output_proj(decoder_output)
            probabilities = F.softmax(output, dim=-1)
            
            # Greedy: pick most likely token
            next_token = probabilities.argmax(dim=-1).item()
            
            # Stop if end token
            if next_token == vocab.get('<EOS>', 1):
                break
            
            decoded_tokens.append(next_token)
            decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
    
    return decoded_tokens

# 5.2 Sampling with Temperature
def sample_with_temperature(model, question, vocab, temperature=1.0, max_length=50):
    """
    Sample from probability distribution with temperature
    
    Formula:
    P'(y_t) = softmax(logits / temperature)
    
    where:
    - temperature < 1: sharper distribution (more confident)
    - temperature > 1: flatter distribution (more diverse)
    - temperature = 1: original distribution
    """
    model.eval()
    device = next(model.parameters()).device
    
    question_tensor = torch.tensor([question], dtype=torch.long).to(device)
    
    with torch.no_grad():
        encoder_embedded = model.embedding(question_tensor)
        encoder_output, (hidden, cell) = model.encoder(encoder_embedded)
        
        decoded_tokens = []
        decoder_input = torch.zeros(1, 1, dtype=torch.long).to(device)
        decoder_hidden = hidden
        decoder_cell = cell
        
        for _ in range(max_length):
            decoder_embedded = model.embedding(decoder_input)
            decoder_output, (decoder_hidden, decoder_cell) = model.decoder(
                decoder_embedded, (decoder_hidden, decoder_cell)
            )
            
            output = model.output_proj(decoder_output)
            
            # Apply temperature
            logits = output / temperature
            probabilities = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            next_token = torch.multinomial(probabilities[0, 0], 1).item()
            
            if next_token == vocab.get('<EOS>', 1):
                break
            
            decoded_tokens.append(next_token)
            decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
    
    return decoded_tokens

# 5.3 Top-k Sampling
def top_k_sampling(model, question, vocab, k=10, max_length=50):
    """
    Top-k Sampling: Sample from top k most likely tokens
    
    Formula:
    1. Get top k tokens: top_k = argsort(P(y_t))[:k]
    2. Renormalize: P'(y_t) = P(y_t) / Σ_{i∈top_k} P(y_i)
    3. Sample from renormalized distribution
    """
    model.eval()
    device = next(model.parameters()).device
    
    question_tensor = torch.tensor([question], dtype=torch.long).to(device)
    
    with torch.no_grad():
        encoder_embedded = model.embedding(question_tensor)
        encoder_output, (hidden, cell) = model.encoder(encoder_embedded)
        
        decoded_tokens = []
        decoder_input = torch.zeros(1, 1, dtype=torch.long).to(device)
        decoder_hidden = hidden
        decoder_cell = cell
        
        for _ in range(max_length):
            decoder_embedded = model.embedding(decoder_input)
            decoder_output, (decoder_hidden, decoder_cell) = model.decoder(
                decoder_embedded, (decoder_hidden, decoder_cell)
            )
            
            output = model.output_proj(decoder_output)
            probabilities = F.softmax(output, dim=-1)
            
            # Get top k
            top_k_probs, top_k_indices = torch.topk(probabilities[0, 0], k)
            
            # Renormalize
            top_k_probs = top_k_probs / top_k_probs.sum()
            
            # Sample
            next_token_idx = torch.multinomial(top_k_probs, 1).item()
            next_token = top_k_indices[next_token_idx].item()
            
            if next_token == vocab.get('<EOS>', 1):
                break
            
            decoded_tokens.append(next_token)
            decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(device)
    
    return decoded_tokens

# 5.4 Convert tokens back to text
def tokens_to_text(tokens, vocab):
    """Convert token indices back to text"""
    idx_to_word = {idx: word for word, idx in vocab.items()}
    words = [idx_to_word.get(token, '<UNK>') for token in tokens]
    return ' '.join(words)

# Example: Generate response
question = "Who won the NBA championship in 2023?"
question_seq = text_to_sequence(preprocess_text(question), vocab)

# Greedy decoding
response_tokens = greedy_decode(model, question_seq, vocab)
response_text = tokens_to_text(response_tokens, vocab)
print(f"Question: {question}")
print(f"Response: {response_text}")

# Temperature sampling (more creative)
response_tokens = sample_with_temperature(model, question_seq, vocab, temperature=0.8)
response_text = tokens_to_text(response_tokens, vocab)
print(f"Response (temperature=0.8): {response_text}")`
    },
    step6: {
      title: 'Step 6: Complete Implementation',
      description: 'Full working chatbot with all components',
      code: `# Step 6: Complete NBA Basketball Chatbot Implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import re
from collections import Counter

class NBAChatbot:
    """
    Complete NBA Basketball Chatbot
    
    Combines all concepts:
    - Linear Algebra: Word embeddings, matrix operations
    - Calculus: Gradient descent, backpropagation
    - Probability: Sampling, softmax distributions
    - Neural Networks: LSTM/Transformer architecture
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
        self.vocab = {}
        self.idx_to_word = {}
        self.model = None
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
    
    def build_vocabulary(self, data):
        """Build vocabulary from NBA data"""
        word_counts = Counter()
        for item in data:
            for word in item['question'].split() + item['answer'].split():
                word_counts[word] += 1
        
        vocab = {word: idx + 2 for idx, (word, count) in enumerate(
            [(w, c) for w, c in word_counts.items() if c >= 2]
        )}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab['<EOS>'] = len(vocab)
        vocab['<SOS>'] = len(vocab)
        
        self.vocab = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.vocab_size = len(vocab)
        
        return vocab
    
    def preprocess_text(self, text):
        """Preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)
        return text
    
    def text_to_sequence(self, text, max_length=50):
        """Convert text to sequence"""
        tokens = self.preprocess_text(text).split()
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        if len(sequence) < max_length:
            sequence = sequence + [self.vocab['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence to text"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in sequence]
        # Remove padding and special tokens
        words = [w for w in words if w not in ['<PAD>', '<EOS>', '<SOS>', '<UNK>']]
        return ' '.join(words)
    
    def train(self, nba_data, epochs=20, batch_size=32, lr=0.001):
        """Train the chatbot"""
        # Build vocabulary
        self.build_vocabulary(nba_data)
        
        # Prepare data
        questions = [self.text_to_sequence(item['question']) for item in nba_data]
        answers = [self.text_to_sequence(item['answer']) for item in nba_data]
        
        # Create dataset
        dataset = list(zip(questions, answers))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMSeq2Seq(
            self.vocab_size, 
            self.embedding_dim, 
            self.hidden_dim
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for questions, answers in train_loader:
                questions = torch.tensor(questions, dtype=torch.long)
                answers = torch.tensor(answers, dtype=torch.long)
                
                decoder_input = answers[:, :-1]
                decoder_target = answers[:, 1:]
                
                predictions = self.model(questions, decoder_input)
                predictions = predictions.reshape(-1, predictions.size(-1))
                decoder_target = decoder_target.reshape(-1)
                
                loss = criterion(predictions, decoder_target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    def chat(self, question, temperature=1.0):
        """Chat with the bot"""
        if self.model is None:
            return "Model not trained yet!"
        
        question_seq = self.text_to_sequence(question)
        question_tensor = torch.tensor([question_seq], dtype=torch.long)
        
        self.model.eval()
        with torch.no_grad():
            response_tokens = sample_with_temperature(
                self.model, question_seq, self.vocab, temperature
            )
        
        response_text = self.sequence_to_text(response_tokens)
        return response_text

# Usage Example
if __name__ == "__main__":
    # NBA data
    nba_data = [
        {"question": "Who won the NBA championship in 2023?", 
         "answer": "The Denver Nuggets won the NBA championship in 2023."},
        {"question": "What team does LeBron James play for?", 
         "answer": "LeBron James plays for the Los Angeles Lakers."},
        {"question": "Who is the all-time leading scorer?", 
         "answer": "LeBron James is the all-time leading scorer in NBA history."},
        # ... more data
    ]
    
    # Initialize chatbot
    chatbot = NBAChatbot()
    
    # Train
    chatbot.train(nba_data, epochs=20)
    
    # Chat
    question = "Who won the NBA championship in 2023?"
    response = chatbot.chat(question)
    print(f"Q: {question}")
    print(f"A: {response}")`
    }
  };

  const pytorchCode = steps[selectedStep]?.code || '';
  const tensorflowCode = pytorchCode.replace(/torch/g, 'tf').replace(/PyTorch/g, 'TensorFlow');

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-6">
        <h2 className="text-2xl font-bold text-purple-900 mb-2">NBA Basketball Chatbot - Complete Tutorial</h2>
        <p className="text-purple-800">
          A comprehensive step-by-step guide covering all mathematical foundations, concepts, theory, and formulas 
          needed to build a chatbot from scratch. This tutorial integrates Linear Algebra, Calculus, Probability, 
          and Neural Networks into a complete working application.
        </p>
      </div>

      <div className="flex gap-4 mb-4">
        <button
          onClick={() => setSelectedFramework('pytorch')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'pytorch'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          PyTorch
        </button>
        <button
          onClick={() => setSelectedFramework('tensorflow')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'tensorflow'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          TensorFlow
        </button>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Tutorial Step
        </label>
        <select
          value={selectedStep}
          onChange={(e) => setSelectedStep(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
        >
          {Object.entries(steps).map(([key, step]) => (
            <option key={key} value={key}>
              {step.title}
            </option>
          ))}
        </select>
      </div>

      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200 mb-4">
        <h3 className="font-semibold text-blue-900 mb-2">{steps[selectedStep]?.title}</h3>
        <p className="text-blue-800 text-sm">{steps[selectedStep]?.description}</p>
      </div>

      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, borderRadius: '0.5rem' }}
          showLineNumbers
        >
          {selectedFramework === 'pytorch' ? pytorchCode : tensorflowCode}
        </SyntaxHighlighter>
      </div>

      <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
        <h3 className="font-semibold text-green-900 mb-2">Concepts Covered in This Step:</h3>
        <ul className="list-disc list-inside space-y-1 text-green-800 text-sm">
          {selectedStep === 'step1' && (
            <>
              <li><strong>Data Preprocessing:</strong> Text cleaning, tokenization, vocabulary building</li>
              <li><strong>Linear Algebra:</strong> Text → Vector representation (sequences)</li>
              <li><strong>Probability:</strong> Word frequency distributions</li>
            </>
          )}
          {selectedStep === 'step2' && (
            <>
              <li><strong>Linear Algebra:</strong> Word embeddings as vectors in high-dimensional space</li>
              <li><strong>Cosine Similarity:</strong> Measuring word similarity using dot products</li>
              <li><strong>Positional Encoding:</strong> Adding position information using trigonometry</li>
            </>
          )}
          {selectedStep === 'step3' && (
            <>
              <li><strong>Neural Networks:</strong> LSTM and Transformer architectures</li>
              <li><strong>Linear Algebra:</strong> Matrix multiplications in neural networks</li>
              <li><strong>Attention Mechanism:</strong> Query-Key-Value attention formula</li>
            </>
          )}
          {selectedStep === 'step4' && (
            <>
              <li><strong>Calculus:</strong> Gradient descent and backpropagation</li>
              <li><strong>Loss Function:</strong> Cross-entropy loss formula</li>
              <li><strong>Optimization:</strong> Adam optimizer with momentum</li>
            </>
          )}
          {selectedStep === 'step5' && (
            <>
              <li><strong>Probability:</strong> Sampling from distributions</li>
              <li><strong>Softmax:</strong> Converting logits to probabilities</li>
              <li><strong>Decoding Strategies:</strong> Greedy, temperature sampling, top-k</li>
            </>
          )}
          {selectedStep === 'step6' && (
            <>
              <li><strong>Complete Integration:</strong> All concepts working together</li>
              <li><strong>End-to-End Pipeline:</strong> From data to working chatbot</li>
              <li><strong>Real-World Application:</strong> Practical implementation</li>
            </>
          )}
        </ul>
      </div>
    </div>
  );
}

