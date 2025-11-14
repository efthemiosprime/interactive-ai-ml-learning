import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ImageSegmentation() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  const pytorchCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 1. U-Net Architecture for Semantic Segmentation
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=21):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        
        # Decoder (upsampling path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        # Output
        logits = self.outc(x)
        return logits

# 2. DeepLabV3 for Semantic Segmentation
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 
                               padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = F.relu(self.bn_out(self.conv_out(out)))
        return out

class DeepLabV3(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet50'):
        super().__init__()
        # Use pre-trained ResNet as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.aspp = ASPP(2048, 256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

# 3. Mask R-CNN for Instance Segmentation
class MaskRCNN(nn.Module):
    """Simplified Mask R-CNN architecture"""
    def __init__(self, n_classes=21):
        super().__init__()
        # Use pre-trained ResNet + FPN as backbone
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        # Replace classifier head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, n_classes
        )
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, n_classes
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

# 4. Loss Function for Segmentation
class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, predictions, targets):
        """
        predictions: [batch, n_classes, height, width]
        targets: [batch, height, width] with class indices
        """
        return self.criterion(predictions, targets)

# 5. Training Function
def train_segmentation_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

# 6. Evaluation Metrics (IoU - Intersection over Union)
def calculate_iou(pred_mask, true_mask, n_classes):
    """Calculate IoU for each class"""
    ious = []
    
    for cls in range(n_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return ious

# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize U-Net model
    model = UNet(n_channels=3, n_classes=21).to(device)
    
    # Example input
    image = torch.randn(1, 3, 256, 256).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image)
        print(f"Output shape: {output.shape}")  # [1, 21, 256, 256]
        
        # Get predicted masks
        predicted_mask = torch.argmax(output, dim=1)
        print(f"Predicted mask shape: {predicted_mask.shape}")  # [1, 256, 256]`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.applications as applications

# 1. U-Net Architecture
def conv_block(input_tensor, n_filters):
    """Double convolution block"""
    x = layers.Conv2D(n_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(n_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def build_unet(input_shape=(256, 256, 3), n_classes=21):
    """Build U-Net model for semantic segmentation"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (downsampling)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = conv_block(p4, 1024)
    
    # Decoder (upsampling) with skip connections
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64)
    
    # Output layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# 2. DeepLabV3 Architecture
def ASPP_block(input_tensor, filters=256):
    """Atrous Spatial Pyramid Pooling"""
    shape = input_tensor.shape
    
    # 1x1 convolution
    x1 = layers.Conv2D(filters, 1, padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    
    # 3x3 convolutions with different dilation rates
    x2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6)(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    
    x3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12)(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    
    x4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18)(input_tensor)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)
    
    # Global average pooling
    x5 = layers.GlobalAveragePooling2D()(input_tensor)
    x5 = layers.Reshape((1, 1, filters))(x5)
    x5 = layers.Conv2D(filters, 1, padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('relu')(x5)
    x5 = layers.UpSampling2D(size=(shape[1], shape[2]), interpolation='bilinear')(x5)
    
    # Concatenate all branches
    out = layers.concatenate([x1, x2, x3, x4, x5])
    out = layers.Conv2D(filters, 1, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    
    return out

def build_deeplabv3(input_shape=(512, 512, 3), n_classes=21):
    """Build DeepLabV3 model"""
    # Use ResNet50 as backbone
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Get feature maps from different levels
    x = base_model.get_layer('conv4_block6_out').output
    
    # Apply ASPP
    x = ASPP_block(x, filters=256)
    
    # Upsample to original size
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Final classification layer
    outputs = layers.Conv2D(n_classes, (1, 1), activation='softmax')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model

# 3. Training Function
def train_segmentation_model(model, train_dataset, val_dataset, epochs=50):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# 4. IoU Metric
class IoU(keras.metrics.Metric):
    def __init__(self, n_classes, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.intersection = self.add_weight(name='intersection', shape=(n_classes,), initializer='zeros')
        self.union = self.add_weight(name='union', shape=(n_classes,), initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        
        for i in range(self.n_classes):
            true_class = tf.cast(y_true == i, tf.float32)
            pred_class = tf.cast(y_pred == i, tf.float32)
            
            intersection = tf.reduce_sum(true_class * pred_class)
            union = tf.reduce_sum(true_class + pred_class - true_class * pred_class)
            
            self.intersection[i].assign_add(intersection)
            self.union[i].assign_add(union)
    
    def result(self):
        iou = self.intersection / (self.union + 1e-7)
        return tf.reduce_mean(iou)

# Example Usage
if __name__ == "__main__":
    # Build U-Net model
    model = build_unet(input_shape=(256, 256, 3), n_classes=21)
    model.summary()
    
    # Compile with IoU metric
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', IoU(n_classes=21)]
    )
    
    # Example prediction
    image = tf.random.normal((1, 256, 256, 3))
    prediction = model.predict(image)
    print(f"Prediction shape: {prediction.shape}")  # [1, 256, 256, 21]
    
    predicted_mask = tf.argmax(prediction, axis=-1)
    print(f"Predicted mask shape: {predicted_mask.shape}")  # [1, 256, 256]`;

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
          <li><strong>Semantic Segmentation:</strong> Classify each pixel into a category</li>
          <li><strong>Instance Segmentation:</strong> Detect and segment individual objects</li>
          <li><strong>U-Net:</strong> Encoder-decoder with skip connections for precise boundaries</li>
          <li><strong>ASPP:</strong> Atrous Spatial Pyramid Pooling captures multi-scale features</li>
          <li><strong>IoU Metric:</strong> Intersection over Union measures segmentation accuracy</li>
        </ul>
      </div>
    </div>
  );
}

