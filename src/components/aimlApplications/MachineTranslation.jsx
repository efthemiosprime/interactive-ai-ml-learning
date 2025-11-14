import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function MachineTranslation() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math

# 1. Positional Encoding for Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)

# 3. Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# 4. Transformer Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention (masked)
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# 5. Complete Transformer Model for Translation
class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        src_emb = self.dropout(src_emb)
        
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Decoder
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.dropout(tgt_emb)
        
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        
        # Output projection
        output = self.output_proj(decoder_output)
        return output

# 6. Training Function
def train_translation_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for src_batch, tgt_batch in train_loader:
        src_batch = src_batch.to(device)
        tgt_input = tgt_batch[:, :-1].to(device)
        tgt_output = tgt_batch[:, 1:].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(src_batch, tgt_input)
        
        # Reshape for loss calculation
        predictions = predictions.view(-1, predictions.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(predictions, tgt_output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TransformerMT(
        src_vocab_size=10000,  # Source language vocabulary
        tgt_vocab_size=10000,  # Target language vocabulary
        d_model=512,
        n_heads=8,
        n_layers=6
    ).to(device)
    
    # Example: Translate "Hello world" to French
    src_sentence = torch.tensor([[1, 2, 3]])  # Tokenized source
    tgt_sentence = torch.tensor([[4, 5]])    # Tokenized target
    
    model.eval()
    with torch.no_grad():
        output = model(src_sentence, tgt_sentence)
        print(f"Translation logits shape: {output.shape}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

# 1. Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         (-math.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

# 2. Multi-Head Attention
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.W_o = layers.Dense(d_model)
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, [batch_size, -1, self.n_heads, self.d_k])
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.reshape(K, [batch_size, -1, self.n_heads, self.d_k])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.reshape(V, [batch_size, -1, self.n_heads, self.d_k])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_output = tf.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, 
                                     [batch_size, -1, self.d_model])
        
        return self.W_o(attention_output)

# 3. Transformer Encoder Block
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, x, mask=None, training=False):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        
        return x

# 4. Transformer Decoder Block
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
    
    def call(self, x, encoder_output, src_mask=None, tgt_mask=None, training=False):
        # Self-attention (masked)
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        
        # Cross-attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output, training=training))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output, training=training))
        
        return x

# 5. Complete Transformer Model
class TransformerMT(keras.Model):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 n_layers=6, d_ff=2048, max_len=5000, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = layers.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Output projection
        self.output_proj = layers.Dense(tgt_vocab_size)
        self.dropout = layers.Dropout(dropout)
    
    def call(self, src, tgt, src_mask=None, tgt_mask=None, training=False):
        # Encoder
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb, training=training)
        
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask, training=training)
        
        # Decoder
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb, training=training)
        
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, 
                                 src_mask, tgt_mask, training=training)
        
        # Output projection
        output = self.output_proj(decoder_output)
        return output

# 6. Training Function
def train_model(model, train_dataset, val_dataset, epochs=10):
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    @tf.function
    def train_step(src, tgt):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        with tf.GradientTape() as tape:
            predictions = model(src, tgt_input, training=True)
            loss = loss_fn(tgt_output, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src_batch, tgt_batch) in enumerate(train_dataset):
            loss = train_step(src_batch, tgt_batch)
            total_loss += loss
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataset):.4f}")

# Example Usage
if __name__ == "__main__":
    # Initialize model
    model = TransformerMT(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Example translation
    src_sentence = tf.constant([[1, 2, 3]])  # Tokenized source
    tgt_sentence = tf.constant([[4, 5]])     # Tokenized target
    
    output = model(src_sentence, tgt_sentence, training=False)
    print(f"Translation logits shape: {output.shape}")`;

  return (
    <div className="space-y-6">
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

      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Key Concepts:</h3>
        <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
          <li><strong>Encoder-Decoder Architecture:</strong> Encodes source language, decodes to target language</li>
          <li><strong>Attention Mechanism:</strong> Allows model to focus on relevant parts of input</li>
          <li><strong>Positional Encoding:</strong> Adds position information to word embeddings</li>
          <li><strong>Multi-Head Attention:</strong> Multiple attention heads capture different relationships</li>
          <li><strong>Beam Search:</strong> Decoding strategy for better translations</li>
        </ul>
      </div>
    </div>
  );
}

