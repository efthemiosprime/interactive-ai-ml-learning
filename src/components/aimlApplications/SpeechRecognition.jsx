import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function SpeechRecognition() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Audio Preprocessing - Extract MFCC Features
class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
        )
    
    def extract_features(self, waveform):
        """Extract MFCC features from audio waveform"""
        # waveform shape: [channels, samples]
        mfcc = self.mfcc_transform(waveform)
        # mfcc shape: [channels, n_mfcc, time]
        return mfcc.squeeze(0)  # Remove channel dimension

# 2. Speech Recognition Model (CNN + RNN)
class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_mfcc=13, n_classes=29, hidden_size=128):
        super().__init__()
        # CNN for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # RNN for sequence modeling
        self.rnn = nn.LSTM(
            input_size=64 * (n_mfcc // 4),
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer (CTC for sequence alignment)
        self.fc = nn.Linear(hidden_size * 2, n_classes)
        
    def forward(self, x):
        # x shape: [batch, channels, n_mfcc, time]
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        
        # Reshape for RNN: [batch, time, features]
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, x.size(1), -1)
        
        # RNN processing
        x, _ = self.rnn(x)
        
        # Output logits for each time step
        x = self.fc(x)
        return x  # [batch, time, n_classes]

# 3. CTC Loss for Sequence Alignment
class CTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        logits: [batch, time, n_classes]
        targets: [batch, max_target_length] - label sequences
        input_lengths: actual sequence lengths
        target_lengths: actual target lengths
        """
        log_probs = nn.functional.log_softmax(logits, dim=2)
        log_probs = log_probs.permute(1, 0, 2)  # [time, batch, n_classes]
        
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss

# 4. Training Loop
def train_speech_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (audio_features, labels, input_lengths, target_lengths) in enumerate(train_loader):
        audio_features = audio_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(audio_features)
        
        # Calculate CTC loss
        loss = criterion(logits, labels, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

# 5. Decoding (Greedy or Beam Search)
def decode_ctc(logits, blank_idx=0):
    """Decode CTC output to text"""
    # Greedy decoding
    predictions = torch.argmax(logits, dim=2)  # [batch, time]
    
    decoded_sequences = []
    for pred in predictions:
        # Remove blanks and consecutive duplicates
        decoded = []
        prev = blank_idx
        for token in pred:
            if token != blank_idx and token != prev:
                decoded.append(token.item())
            prev = token
        decoded_sequences.append(decoded)
    
    return decoded_sequences

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SpeechRecognitionModel(n_mfcc=13, n_classes=29).to(device)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Load audio file
    waveform, sample_rate = torchaudio.load('audio.wav')
    features = preprocessor.extract_features(waveform)
    
    # Prepare input (add batch and channel dimensions)
    features = features.unsqueeze(0).unsqueeze(0)  # [1, 1, n_mfcc, time]
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(features)
        decoded = decode_ctc(logits)
        print(f"Decoded text: {decoded}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import numpy as np

# 1. Audio Preprocessing - Extract MFCC Features
class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=400,
            hop_length=160,
            n_mels=23
        )
        
        return mfcc.T  # [time, n_mfcc]

# 2. Speech Recognition Model (CNN + RNN)
def build_speech_model(n_mfcc=13, n_classes=29, hidden_size=128):
    model = keras.Sequential([
        # Reshape for CNN: [batch, time, n_mfcc, 1]
        layers.Input(shape=(None, n_mfcc)),
        layers.Reshape((-1, n_mfcc, 1)),
        
        # CNN layers for feature extraction
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Reshape for RNN: [batch, time, features]
        layers.Reshape((-1, 64 * (n_mfcc // 4))),
        
        # Bidirectional LSTM
        layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True)),
        
        # Dense layer for output
        layers.Dense(n_classes, activation='softmax')
    ])
    
    return model

# 3. CTC Loss Function
def ctc_loss_fn(y_true, y_pred):
    """CTC loss for sequence alignment"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    loss = tf.keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )
    return loss

# 4. CTC Decoder Layer
class CTCDecoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # Greedy decoding
        input_length = tf.shape(inputs)[1]
        predictions = tf.argmax(inputs, axis=2)
        
        # Remove blanks and duplicates
        decoded = tf.map_fn(
            lambda x: self._remove_blanks(x),
            predictions,
            fn_output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )
        
        return decoded
    
    def _remove_blanks(self, sequence):
        # Remove blank tokens (0) and consecutive duplicates
        mask = tf.not_equal(sequence, 0)
        filtered = tf.boolean_mask(sequence, mask)
        
        # Remove consecutive duplicates
        shifted = tf.concat([[filtered[0]], filtered[:-1]], axis=0)
        unique_mask = tf.not_equal(filtered, shifted)
        return tf.boolean_mask(filtered, unique_mask)

# 5. Complete Model with CTC
def build_ctc_model(n_mfcc=13, n_classes=29):
    inputs = layers.Input(shape=(None, n_mfcc), name="input")
    
    # Feature extraction
    x = layers.Reshape((-1, n_mfcc, 1))(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RNN
    x = layers.Reshape((-1, 64 * (n_mfcc // 4)))(x)
    
    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, activation='softmax', name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with CTC loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=ctc_loss_fn,
        metrics=['accuracy']
    )
    
    return model

# 6. Training
def train_model(model, train_data, val_data, epochs=10):
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# Example Usage
if __name__ == "__main__":
    # Build model
    model = build_ctc_model(n_mfcc=13, n_classes=29)
    model.summary()
    
    # Preprocess audio
    preprocessor = AudioPreprocessor()
    features = preprocessor.extract_features('audio.wav')
    
    # Prepare input (add batch dimension)
    features = np.expand_dims(features, axis=0)  # [1, time, n_mfcc]
    
    # Inference
    predictions = model.predict(features)
    
    # Decode
    decoder = CTCDecoder()
    decoded = decoder(predictions)
    print(f"Decoded text: {decoded}")`;

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
          <li><strong>MFCC Features:</strong> Mel-frequency cepstral coefficients capture audio characteristics</li>
          <li><strong>CNN Layers:</strong> Extract spatial features from spectrograms</li>
          <li><strong>RNN/LSTM:</strong> Model temporal dependencies in speech sequences</li>
          <li><strong>CTC Loss:</strong> Connectionist Temporal Classification aligns sequences without explicit alignment</li>
          <li><strong>Beam Search:</strong> Advanced decoding strategy for better accuracy</li>
        </ul>
      </div>
    </div>
  );
}

