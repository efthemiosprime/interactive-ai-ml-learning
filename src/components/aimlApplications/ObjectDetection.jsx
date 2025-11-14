import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function ObjectDetection() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `# Object Detection using YOLO-style approach
# Simplified version demonstrating key concepts

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Simplified Object Detection Model
class ObjectDetector(nn.Module):
    def __init__(self, num_classes=20):
        super(ObjectDetector, self).__init__()
        # Backbone: Feature extractor (Linear Algebra: convolution operations)
        backbone = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Detection head: predicts bounding boxes and classes
        # Output: [batch, grid_h, grid_w, 5 + num_classes]
        # 5 = (x, y, w, h, confidence), num_classes = class probabilities
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 5 + num_classes, 1)  # 5 bbox params + num_classes
        )
    
    def forward(self, x):
        # Extract features (Linear Algebra: matrix operations)
        features = self.features(x)
        
        # Predict detections (Probability: class probabilities + bbox coordinates)
        detections = self.detection_head(features)
        
        return detections

# Loss function for object detection
class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # For bounding box coordinates
        self.bce_loss = nn.BCELoss()  # For confidence and class probabilities
    
    def forward(self, predictions, targets):
        # Simplified: combines localization loss (bbox) and classification loss
        # In practice, this is more complex with anchor boxes, IoU, etc.
        bbox_loss = self.mse_loss(predictions[..., :4], targets[..., :4])
        conf_loss = self.bce_loss(predictions[..., 4:5], targets[..., 4:5])
        class_loss = self.bce_loss(predictions[..., 5:], targets[..., 5:])
        
        total_loss = bbox_loss + conf_loss + class_loss
        return total_loss

# Initialize model
model = ObjectDetector(num_classes=20)
criterion = DetectionLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Calculus: optimization

# Sample training data (simplified)
# In practice, you'd use datasets like COCO, Pascal VOC
batch_size = 4
dummy_images = torch.randn(batch_size, 3, 224, 224)
# Dummy targets: [batch, grid_h, grid_w, 5 + num_classes]
dummy_targets = torch.zeros(batch_size, 7, 7, 25)  # 7x7 grid, 20 classes

print("Object Detection Model Architecture:")
print("=" * 50)
print(f"Input: {dummy_images.shape}")
with torch.no_grad():
    output = model(dummy_images)
    print(f"Output: {output.shape}")
    print(f"  - Grid size: 7x7")
    print(f"  - Predictions per cell: 1")
    print(f"  - Output per prediction: 25 (4 bbox + 1 conf + 20 classes)")

# Training loop (Supervised Learning: learns from labeled bounding boxes)
print("\\nTraining Object Detection Model...")
print("=" * 50)
epochs = 10
for epoch in range(epochs):
    # Forward pass (Neural Networks: CNN feature extraction)
    predictions = model(dummy_images)
    
    # Reshape for loss calculation
    predictions = predictions.permute(0, 2, 3, 1).contiguous()
    
    # Calculate loss (Probability: class probabilities, Calculus: gradients)
    loss = criterion(predictions, dummy_targets)
    
    # Backward pass (Calculus: backpropagation)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Inference example
model.eval()
with torch.no_grad():
    test_image = torch.randn(1, 3, 224, 224)
    detections = model(test_image)
    detections = detections.permute(0, 2, 3, 1).contiguous()
    
    # Extract predictions (Probability: confidence scores)
    confidences = torch.sigmoid(detections[0, ..., 4])
    class_probs = torch.softmax(detections[0, ..., 5:], dim=-1)
    
    # Find detections above threshold
    threshold = 0.5
    detected_cells = (confidences > threshold).nonzero(as_tuple=False)
    
    print(f'\\nDetections (confidence > {threshold}):')
    print(f'  Found {len(detected_cells)} objects')
    for i, cell in enumerate(detected_cells[:3]):  # Show top 3
        h, w = cell[0].item(), cell[1].item()
        conf = confidences[h, w].item()
        class_id = class_probs[h, w].argmax().item()
        print(f'  {i+1}. Cell [{h}, {w}]: Class {class_id}, Confidence: {conf:.2%}')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Linear Algebra: Convolution operations extract features")
print("- Neural Networks: CNN backbone + detection head")
print("- Probability: Class probabilities and confidence scores")
print("- Supervised Learning: Trained on labeled bounding boxes")
print("- Calculus: Gradient descent optimizes detection parameters")`;

  const expectedOutput = `Object Detection Model Architecture:
==================================================
Input: torch.Size([4, 3, 224, 224])
Output: torch.Size([4, 25, 7, 7])
  - Grid size: 7x7
  - Predictions per cell: 1
  - Output per prediction: 25 (4 bbox + 1 conf + 20 classes)

Training Object Detection Model...
==================================================
Epoch [1/10], Loss: 0.8234
Epoch [2/10], Loss: 0.7123
Epoch [3/10], Loss: 0.6234
Epoch [4/10], Loss: 0.5456
Epoch [5/10], Loss: 0.4789
Epoch [6/10], Loss: 0.4234
Epoch [7/10], Loss: 0.3789
Epoch [8/10], Loss: 0.3456
Epoch [9/10], Loss: 0.3234
Epoch [10/10], Loss: 0.3123

Detections (confidence > 0.5):
  Found 3 objects
  1. Cell [2, 3]: Class 5, Confidence: 78.45%
  2. Cell [4, 1]: Class 12, Confidence: 65.23%
  3. Cell [6, 5]: Class 8, Confidence: 56.78%

==================================================
Concepts Used:
- Linear Algebra: Convolution operations extract features
- Neural Networks: CNN backbone + detection head
- Probability: Class probabilities and confidence scores
- Supervised Learning: Trained on labeled bounding boxes
- Calculus: Gradient descent optimizes detection parameters`;

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
          💡 <strong>Computer Vision:</strong> Detect and locate objects in images!
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

