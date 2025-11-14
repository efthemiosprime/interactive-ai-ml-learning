import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function PretrainedModels({ framework }) {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const pytorchCode = `import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Example: Load and preprocess an image
# image = Image.open('path/to/image.jpg')
# input_tensor = preprocess(image).unsqueeze(0)

# For demo, create a dummy image tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
# Get top 5 predictions
top5_prob, top5_idx = torch.topk(probabilities, 5)
print("Top 5 predictions:")
for i in range(5):
    print(f"  {i+1}. Class {top5_idx[i].item()}: {top5_prob[i].item():.2%}")`;

  const tensorflowCode = `import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load pre-trained MobileNetV2 model
model = keras.applications.MobileNetV2(
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Image preprocessing
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

# Example: Load and preprocess an image
# image = keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
# image_array = keras.preprocessing.image.img_to_array(image)
# image_array = np.expand_dims(image_array, axis=0)
# image_array = preprocess_input(image_array)

# For demo, create a dummy image array
image_array = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
image_array = preprocess_input(image_array.astype(np.float32))

# Inference
predictions = model.predict(image_array, verbose=0)
decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

print("Top 5 predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"  {i+1}. {label}: {score:.2%}")`;

  const code = framework === 'pytorch' ? pytorchCode : tensorflowCode;
  const expectedOutput = framework === 'pytorch'
    ? `Top 5 predictions:
  1. Class 285: 12.34%
  2. Class 123: 8.90%
  3. Class 456: 7.65%
  4. Class 789: 6.23%
  5. Class 234: 5.67%`
    : `Top 5 predictions:
  1. Egyptian_cat: 12.34%
  2. tabby: 8.90%
  3. tiger_cat: 7.65%
  4. lynx: 6.23%
  5. Persian_cat: 5.67%`;

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
      <div className="bg-orange-50 border-2 border-orange-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-orange-800">
          💡 <strong>Pre-trained Models:</strong> Use models trained on ImageNet for image classification!
        </p>
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
        <pre className="text-green-400 text-sm overflow-x-auto">
          <code>{code}</code>
        </pre>
      </div>

      {/* Run Button */}
      <button
        onClick={runCode}
        className="w-full px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-semibold flex items-center justify-center gap-2"
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

      {/* Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">Benefits of Pre-trained Models:</h4>
        <ul className="list-disc list-inside space-y-2 text-sm text-blue-800">
          <li><strong>Transfer Learning:</strong> Use knowledge from large datasets (ImageNet)</li>
          <li><strong>Time Saving:</strong> No need to train from scratch</li>
          <li><strong>Better Performance:</strong> Trained on millions of images</li>
          <li><strong>Fine-tuning:</strong> Adapt to your specific task with minimal training</li>
        </ul>
      </div>
    </div>
  );
}

