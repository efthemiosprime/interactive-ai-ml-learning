import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function TransferLearning() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 1. Feature Extraction Approach (Freeze Backbone)
def create_feature_extractor(num_classes=10):
    """Use pre-trained model as feature extractor"""
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# 2. Fine-tuning Approach (Unfreeze Some Layers)
def create_finetuned_model(num_classes=10, freeze_backbone=True):
    """Fine-tune pre-trained model"""
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    if freeze_backbone:
        # Freeze early layers, keep later layers trainable
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    else:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

# 3. Transfer Learning with Different Architectures
class CustomClassifier(nn.Module):
    """Custom classifier on top of pre-trained features"""
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def create_transfer_model(backbone_name='resnet50', num_classes=10):
    """Create transfer learning model with different backbones"""
    # Available backbones
    backbones = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'vgg16': models.vgg16,
        'efficientnet': models.efficientnet_b0,
        'mobilenet': models.mobilenet_v2
    }
    
    # Load pre-trained backbone
    backbone = backbones[backbone_name](pretrained=True)
    
    # Extract feature dimension
    if 'resnet' in backbone_name:
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove classifier
    elif 'vgg' in backbone_name:
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    elif 'efficientnet' in backbone_name:
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif 'mobilenet' in backbone_name:
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    
    # Add custom classifier
    classifier = CustomClassifier(feature_dim, num_classes)
    
    # Combine
    model = nn.Sequential(backbone, classifier)
    
    return model

# 4. Domain Adaptation (Different Domains)
class DomainAdaptationModel(nn.Module):
    """Model for domain adaptation"""
    def __init__(self, backbone, num_classes, feature_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        # Task classifier (source domain)
        self.task_classifier = nn.Linear(feature_dim, num_classes)
        
        # Domain classifier (distinguish source/target)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Source vs Target
        )
    
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.backbone(x)
        
        # Task prediction
        task_output = self.task_classifier(features)
        
        # Domain prediction (with gradient reversal)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return task_output, domain_output

class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# 5. Multi-task Learning
class MultiTaskModel(nn.Module):
    """Model for multiple related tasks"""
    def __init__(self, backbone, num_classes_task1, num_classes_task2):
        super().__init__()
        self.backbone = backbone
        
        # Shared feature extractor
        if hasattr(backbone, 'fc'):
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            feature_dim = 2048
        
        # Task-specific heads
        self.task1_head = nn.Linear(feature_dim, num_classes_task1)
        self.task2_head = nn.Linear(feature_dim, num_classes_task2)
    
    def forward(self, x):
        features = self.backbone(x)
        output1 = self.task1_head(features)
        output2 = self.task2_head(features)
        return output1, output2

# 6. Training with Transfer Learning
def train_transfer_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Train transfer learning model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use different learning rates for different parts
    optimizer = optim.SGD([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': model.classifier.parameters(), 'lr': lr}        # Higher LR for new layers
    ], momentum=0.9, weight_decay=0.0001)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# 7. Using Pre-trained Transformers (NLP)
from transformers import AutoModel, AutoTokenizer

class TextClassifier(nn.Module):
    """Text classifier using pre-trained BERT"""
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Example Usage
if __name__ == "__main__":
    # Feature extraction approach
    model1 = create_feature_extractor(num_classes=10)
    print("Feature extractor model created")
    
    # Fine-tuning approach
    model2 = create_finetuned_model(num_classes=10, freeze_backbone=True)
    print("Fine-tuned model created")
    
    # Transfer with different backbone
    model3 = create_transfer_model(backbone_name='efficientnet', num_classes=10)
    print("Transfer model with EfficientNet created")
    
    # Example: Image classification
    image = torch.randn(1, 3, 224, 224)
    output = model2(image)
    print(f"Output shape: {output.shape}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np

# 1. Feature Extraction Approach
def create_feature_extractor(input_shape=(224, 224, 3), num_classes=10):
    """Use pre-trained model as feature extractor"""
    # Load pre-trained ResNet50
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 2. Fine-tuning Approach
def create_finetuned_model(input_shape=(224, 224, 3), num_classes=10, 
                          freeze_backbone=True):
    """Fine-tune pre-trained model"""
    # Load pre-trained ResNet50
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    if freeze_backbone:
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    else:
        # Unfreeze all layers
        base_model.trainable = True
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 3. Transfer Learning with Different Architectures
def create_transfer_model(backbone_name='ResNet50', num_classes=10):
    """Create transfer learning model with different backbones"""
    backbones = {
        'ResNet50': applications.ResNet50,
        'ResNet101': applications.ResNet101,
        'VGG16': applications.VGG16,
        'EfficientNetB0': applications.EfficientNetB0,
        'MobileNetV2': applications.MobileNetV2,
        'InceptionV3': applications.InceptionV3
    }
    
    # Load pre-trained backbone
    base_model = backbones[backbone_name](
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 4. Domain Adaptation
class DomainAdaptationModel(keras.Model):
    """Model for domain adaptation"""
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.task_classifier = layers.Dense(num_classes, activation='softmax')
        self.domain_classifier = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # Source vs Target
        ])
    
    def call(self, x, training=False):
        features = self.base_model(x, training=training)
        task_output = self.task_classifier(features)
        domain_output = self.domain_classifier(features)
        return task_output, domain_output

# 5. Multi-task Learning
class MultiTaskModel(keras.Model):
    """Model for multiple related tasks"""
    def __init__(self, base_model, num_classes_task1, num_classes_task2):
        super().__init__()
        self.base_model = base_model
        self.task1_head = layers.Dense(num_classes_task1, activation='softmax')
        self.task2_head = layers.Dense(num_classes_task2, activation='softmax')
    
    def call(self, x, training=False):
        features = self.base_model(x, training=training)
        output1 = self.task1_head(features)
        output2 = self.task2_head(features)
        return output1, output2

# 6. Training with Transfer Learning
def train_transfer_model(model, train_dataset, val_dataset, epochs=10):
    """Train transfer learning model"""
    # Compile with different learning rates for different parts
    # Lower LR for pre-trained layers, higher for new layers
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=0.001,
            momentum=0.9
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history, model

# 7. Using Pre-trained Transformers (NLP)
from transformers import TFAutoModel, AutoTokenizer

class TextClassifier(keras.Model):
    """Text classifier using pre-trained BERT"""
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        self.bert = TFAutoModel.from_pretrained(model_name)
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, input_ids, attention_mask, training=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                           training=training)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output, training=training)
        return self.classifier(output)

# 8. Progressive Unfreezing Strategy
def train_with_progressive_unfreezing(model, train_dataset, val_dataset, epochs=30):
    """Train with progressive unfreezing"""
    # Stage 1: Train only classifier
    model.base_model.trainable = False
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
    # Stage 2: Unfreeze last few layers
    for layer in model.base_model.layers[-10:]:
        layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
    # Stage 3: Unfreeze all layers with lower learning rate
    model.base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    
    return model

# Example Usage
if __name__ == "__main__":
    # Feature extraction
    model1 = create_feature_extractor(num_classes=10)
    print("Feature extractor model created")
    
    # Fine-tuning
    model2 = create_finetuned_model(num_classes=10, freeze_backbone=True)
    print("Fine-tuned model created")
    
    # Transfer with different backbone
    model3 = create_transfer_model(backbone_name='EfficientNetB0', num_classes=10)
    print("Transfer model with EfficientNet created")
    
    # Example: Image classification
    image = tf.random.normal((1, 224, 224, 3))
    output = model2(image)
    print(f"Output shape: {output.shape}")`;

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
          <li><strong>Feature Extraction:</strong> Use pre-trained model as fixed feature extractor</li>
          <li><strong>Fine-tuning:</strong> Unfreeze some layers and train on new data</li>
          <li><strong>Progressive Unfreezing:</strong> Gradually unfreeze layers during training</li>
          <li><strong>Domain Adaptation:</strong> Adapt model to different data distributions</li>
          <li><strong>Multi-task Learning:</strong> Share features across related tasks</li>
        </ul>
      </div>
    </div>
  );
}

