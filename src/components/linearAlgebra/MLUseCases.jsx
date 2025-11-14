import React, { useState } from 'react';
import { Brain, Network, Image, MessageSquare, TrendingUp, Layers, Zap } from 'lucide-react';

export default function MLUseCases({ operationType }) {
  const useCases = {
    'matrix-multiplication': [
      {
        title: 'Neural Network Forward Propagation',
        icon: <Network className="w-6 h-6" />,
        description: 'Each layer multiplies input by weight matrix to produce output',
        example: 'In a fully connected layer: output = activation(W × input + bias)',
        code: `# PyTorch example
output = torch.matmul(input, weight_matrix)
output = activation(output + bias)`,
        realWorld: 'Image classification, natural language processing, recommendation systems'
      },
      {
        title: 'Batch Processing',
        icon: <Layers className="w-6 h-6" />,
        description: 'Process multiple samples simultaneously for efficiency',
        example: 'Process 32 images at once: Batch(32×784) × Weights(784×128) = Output(32×128)',
        code: `# Process batch of 32 images, each with 784 features
batch = torch.randn(32, 784)  # 32 samples, 784 features
weights = torch.randn(784, 128)  # 784 inputs, 128 outputs
output = torch.matmul(batch, weights)  # Result: (32, 128)`,
        realWorld: 'Training neural networks faster, real-time inference in production'
      },
      {
        title: 'Attention Mechanisms (Transformers)',
        icon: <Brain className="w-6 h-6" />,
        description: 'Query × Key^T computes similarity scores between tokens',
        example: 'Attention(Q, K, V) = softmax(Q × K^T / √d) × V',
        code: `# Self-attention in transformers
scores = torch.matmul(query, key.transpose(-2, -1))
attention_weights = F.softmax(scores / math.sqrt(d_k), dim=-1)
output = torch.matmul(attention_weights, value)`,
        realWorld: 'ChatGPT, BERT, GPT models - understanding context in text'
      },
      {
        title: 'Convolutional Neural Networks',
        icon: <Image className="w-6 h-6" />,
        description: 'Filter matrices slide over image patches',
        example: 'Each convolution multiplies filter weights with image patch',
        code: `# Convolution operation
# Filter: (out_channels, in_channels, kernel_h, kernel_w)
# Input: (batch, in_channels, height, width)
output = F.conv2d(input, filters)  # Uses matrix multiplication internally`,
        realWorld: 'Image recognition, medical imaging, autonomous vehicles'
      }
    ],
    'matrix-addition': [
      {
        title: 'Bias Addition',
        icon: <TrendingUp className="w-6 h-6" />,
        description: 'Add bias vector to each neuron output',
        example: 'output = W × input + b, where b shifts the activation function',
        code: `# Adding bias to layer output
output = torch.matmul(input, weights) + bias
# Bias shape: (out_features,) - broadcasted to each sample`,
        realWorld: 'Every neural network layer uses bias to adjust activation thresholds'
      },
      {
        title: 'Residual Connections (ResNet)',
        icon: <Network className="w-6 h-6" />,
        description: 'Add input to output: output = F(x) + x',
        example: 'Helps with gradient flow in deep networks',
        code: `# Residual block
def residual_block(x):
    out = conv_layer(x)
    return out + x  # Element-wise addition`,
        realWorld: 'Deep learning architectures, image classification, object detection'
      }
    ],
    'transpose': [
      {
        title: 'Backpropagation',
        icon: <Brain className="w-6 h-6" />,
        description: 'Transpose weight matrix to propagate gradients backward',
        example: 'grad_input = grad_output × W^T',
        code: `# Backward pass
grad_input = torch.matmul(grad_output, weight.t())
# Transpose needed to match dimensions for gradient flow`,
        realWorld: 'Training all neural networks - gradient descent optimization'
      },
      {
        title: 'Attention Mechanism',
        icon: <MessageSquare className="w-6 h-6" />,
        description: 'Key^T used in attention score calculation',
        example: 'scores = Q × K^T to compute pairwise similarities',
        code: `# Attention scores
scores = torch.matmul(query, key.transpose(-2, -1))
# Transpose key to compute dot products between query and key vectors`,
        realWorld: 'Transformer models, BERT, GPT, language understanding'
      }
    ],
    'element-wise': [
      {
        title: 'Activation Functions',
        icon: <TrendingUp className="w-6 h-6" />,
        description: 'Apply non-linear function element-wise to each value',
        example: 'ReLU: output = max(0, input), applied to each element',
        code: `# ReLU activation
output = F.relu(input)  # Element-wise: max(0, x) for each x
# Sigmoid: 1 / (1 + exp(-x))
# Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))`,
        realWorld: 'All neural networks - introduces non-linearity for complex patterns'
      },
      {
        title: 'Normalization Layers',
        icon: <Layers className="w-6 h-6" />,
        description: 'Normalize each feature independently',
        example: 'BatchNorm: normalize across batch dimension',
        code: `# Batch normalization
normalized = (x - mean) / sqrt(variance + eps)
output = gamma * normalized + beta  # Element-wise operations`,
        realWorld: 'Stable training, faster convergence, better performance'
      }
    ],
    'eigenvalues': [
      {
        title: 'Principal Component Analysis (PCA)',
        icon: <TrendingUp className="w-6 h-6" />,
        description: 'Eigenvalues determine which components capture most variance',
        example: 'Largest eigenvalues = most important features',
        code: `# PCA using eigenvalues
covariance_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# Sort by eigenvalues, select top k components
top_k = eigenvectors[:, np.argsort(eigenvalues)[-k:]]`,
        realWorld: 'Dimensionality reduction, feature extraction, data visualization'
      },
      {
        title: 'Data Compression',
        icon: <Image className="w-6 h-6" />,
        description: 'Eigenvectors define directions of maximum variance',
        example: 'Keep only top eigenvectors to compress data',
        code: `# Compress images using PCA
# Original: (height, width)
# Compressed: (height, k) where k << width
compressed = image @ top_eigenvectors`,
        realWorld: 'Image compression, feature engineering, reducing overfitting'
      }
    ],
    'data-representation': [
      {
        title: 'Image Classification',
        icon: <Image className="w-6 h-6" />,
        description: 'Images flattened into feature vectors (rows = pixels, columns = RGB)',
        example: '28×28 image → 784×1 vector → reshaped to 1×784 for batch processing',
        code: `# MNIST digit classification
# Each image: 28×28 = 784 pixels
# Batch of 32 images: (32, 784) matrix
images = torch.randn(32, 784)  # 32 samples, 784 features
labels = torch.randint(0, 10, (32,))  # 32 labels`,
        realWorld: 'Handwriting recognition, medical imaging, autonomous vehicles'
      },
      {
        title: 'Natural Language Processing',
        icon: <MessageSquare className="w-6 h-6" />,
        description: 'Words encoded as vectors, sentences as matrices',
        example: 'Each word = embedding vector, sentence = sequence of vectors',
        code: `# Word embeddings
# Vocabulary size: 50,000 words
# Embedding dimension: 300
embeddings = nn.Embedding(50000, 300)
# Sentence: (batch_size, seq_length, embedding_dim)
sentence = embeddings(word_ids)  # (32, 50, 300)`,
        realWorld: 'ChatGPT, translation, sentiment analysis, chatbots'
      },
      {
        title: 'Tabular Data Processing',
        icon: <Layers className="w-6 h-6" />,
        description: 'Each row is a sample, each column is a feature',
        example: 'Customer data: rows = customers, columns = age, income, purchases, etc.',
        code: `# Customer churn prediction
# Features: age, income, num_purchases, days_since_last_purchase
data = np.array([
    [25, 50000, 10, 5],   # Customer 1
    [35, 75000, 25, 2],   # Customer 2
    [45, 100000, 50, 1]   # Customer 3
])  # Shape: (3, 4) - 3 samples, 4 features`,
        realWorld: 'Fraud detection, recommendation systems, credit scoring'
      }
    ],
    'weight-representation': [
      {
        title: 'Fully Connected Layers',
        icon: <Network className="w-6 h-6" />,
        description: 'Weight matrix connects every input neuron to every output neuron',
        example: 'Input: 784 neurons, Output: 128 neurons → Weight matrix: 784×128',
        code: `# Linear layer in PyTorch
layer = nn.Linear(784, 128)
# Weight matrix shape: (128, 784)
# Bias vector shape: (128,)
# Total parameters: 784×128 + 128 = 100,480`,
        realWorld: 'Every neural network layer - image classification, NLP, all ML models'
      },
      {
        title: 'Multi-Layer Networks',
        icon: <Layers className="w-6 h-6" />,
        description: 'Each layer has its own weight matrix',
        example: '784 → 256 → 128 → 10: Three weight matrices connecting layers',
        code: `# Multi-layer network
model = nn.Sequential(
    nn.Linear(784, 256),  # W1: (256, 784)
    nn.ReLU(),
    nn.Linear(256, 128),  # W2: (128, 256)
    nn.ReLU(),
    nn.Linear(128, 10)    # W3: (10, 128)
)
# Total parameters: 784×256 + 256×128 + 128×10 = ~250K`,
        realWorld: 'Deep learning models, computer vision, natural language processing'
      },
      {
        title: 'Transfer Learning',
        icon: <Brain className="w-6 h-6" />,
        description: 'Pre-trained weight matrices reused for new tasks',
        example: 'ImageNet weights → fine-tune last layer for custom classification',
        code: `# Transfer learning
pretrained_model = models.resnet50(pretrained=True)
# Freeze early layers, retrain last layer
for param in pretrained_model.parameters():
    param.requires_grad = False
# Replace last layer
pretrained_model.fc = nn.Linear(2048, num_classes)`,
        realWorld: 'Medical imaging, custom image classifiers, domain adaptation'
      }
    ],
    'determinant': [
      {
        title: 'Matrix Invertibility Check',
        icon: <Network className="w-6 h-6" />,
        description: 'Determinant determines if a matrix can be inverted (det ≠ 0)',
        example: 'If det(A) = 0, matrix is singular and cannot be inverted',
        code: `# Check if matrix is invertible
det = np.linalg.det(matrix)
if abs(det) < 1e-10:
    print("Matrix is singular, cannot invert")
else:
    inverse = np.linalg.inv(matrix)`,
        realWorld: 'Solving linear systems, computing matrix inverses, checking numerical stability'
      },
      {
        title: 'Linear Independence Detection',
        icon: <TrendingUp className="w-6 h-6" />,
        description: 'Zero determinant indicates linearly dependent columns/rows',
        example: 'det(A) = 0 means columns are linearly dependent (redundant features)',
        code: `# Check for linear dependence in feature matrix
det = np.linalg.det(covariance_matrix)
if det == 0:
    print("Features are linearly dependent - remove redundant features")
    # Use PCA or feature selection to remove dependencies`,
        realWorld: 'Feature engineering, dimensionality reduction, removing multicollinearity'
      },
      {
        title: 'Volume/Area Scaling in Transformations',
        icon: <Image className="w-6 h-6" />,
        description: 'Absolute determinant = scaling factor for volumes/areas',
        example: 'If det(A) = 2, transformation doubles the area/volume',
        code: `# Calculate volume scaling factor
transformation_matrix = np.array([[2, 0], [0, 3]])  # Scale x by 2, y by 3
scaling_factor = abs(np.linalg.det(transformation_matrix))  # = 6
# Area is scaled by factor of 6`,
        realWorld: 'Image transformations, data augmentation, geometric operations'
      },
      {
        title: 'Jacobian Determinant in Change of Variables',
        icon: <Brain className="w-6 h-6" />,
        description: 'Jacobian determinant used in probability density transformations',
        example: 'When transforming probability distributions, det(J) adjusts density',
        code: `# Change of variables in probability distributions
# Transform x to y: y = f(x)
# New density: p_y(y) = p_x(x) / |det(J)|
# where J is Jacobian matrix of transformation
jacobian = compute_jacobian(transformation_function)
det_j = np.linalg.det(jacobian)
new_density = old_density / abs(det_j)`,
        realWorld: 'Normalizing flows, variational autoencoders, probabilistic models'
      },
      {
        title: 'Covariance Matrix Determinant',
        icon: <Layers className="w-6 h-6" />,
        description: 'Determinant of covariance matrix measures spread of multivariate data',
        example: 'Larger det(Σ) = more spread out data, smaller = more concentrated',
        code: `# Measure data spread using covariance determinant
cov_matrix = np.cov(data.T)
det_cov = np.linalg.det(cov_matrix)
# Large det_cov = data is spread out
# Small det_cov = data is concentrated`,
        realWorld: 'Multivariate analysis, anomaly detection, understanding data distribution'
      },
      {
        title: 'Numerical Stability in Optimization',
        icon: <MessageSquare className="w-6 h-6" />,
        description: 'Near-zero determinants indicate numerical issues in optimization',
        example: 'If det(Hessian) ≈ 0, optimization may be unstable',
        code: `# Check numerical stability during optimization
hessian = compute_hessian(loss_function, parameters)
det_hessian = np.linalg.det(hessian)
if abs(det_hessian) < 1e-10:
    print("Warning: Near-singular Hessian - optimization may be unstable")
    # Use regularization or different optimizer`,
        realWorld: 'Training neural networks, second-order optimization methods, numerical analysis'
      }
    ],
    'matrix-transformations': [
      {
        title: 'Data Augmentation for Image Classification',
        icon: <Image className="w-6 h-6" />,
        description: 'Apply rotation, translation, scaling to increase training data diversity',
        example: 'Rotate images by 15°, translate by 5 pixels, scale by 1.1x to create new training samples',
        code: `# Image augmentation using transformations
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(15),      # Rotation matrix
    transforms.RandomAffine(translate=(0.1, 0.1)),  # Translation
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # Scaling
])
augmented_image = transform(image)`,
        realWorld: 'Image classification, object detection, medical imaging - prevents overfitting'
      },
      {
        title: 'Computer Vision: Object Detection',
        icon: <Network className="w-6 h-6" />,
        description: 'Transform bounding boxes when images are rotated or scaled',
        example: 'When image rotates 30°, transform bounding box coordinates using rotation matrix',
        code: `# Transform bounding box coordinates
import cv2
import numpy as np

# Rotation matrix
angle = 30
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Transform bounding box corners
corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix)`,
        realWorld: 'Autonomous vehicles, surveillance, robotics - handle rotated/scaled objects'
      },
      {
        title: '3D Graphics and Rendering',
        icon: <Layers className="w-6 h-6" />,
        description: 'Transform 3D objects using rotation, translation, scaling matrices',
        example: 'Rotate 3D model around Y-axis, translate camera position, scale object size',
        code: `# 3D transformation pipeline
import numpy as np

# Rotation around Y-axis
def rotation_y(angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([
        [cos_a, 0, sin_a, 0],
        [0, 1, 0, 0],
        [-sin_a, 0, cos_a, 0],
        [0, 0, 0, 1]
    ])

# Combine transformations
model_matrix = translation @ rotation_y(angle) @ scaling`,
        realWorld: 'Video games, virtual reality, 3D modeling, computer graphics'
      },
      {
        title: 'Neural Network Weight Initialization',
        icon: <Brain className="w-6 h-6" />,
        description: 'Use rotation matrices for orthogonal weight initialization',
        example: 'Initialize weights as orthogonal matrices (rotation matrices) for better training',
        code: `# Orthogonal initialization (rotation matrices)
import torch.nn.init as init

# Initialize weights as orthogonal matrices
# Orthogonal matrices preserve vector norms and angles
weight = torch.empty(128, 128)
init.orthogonal_(weight)  # Fills with orthogonal matrix (rotation/reflection)`,
        realWorld: 'Deep learning initialization, preventing vanishing/exploding gradients'
      },
      {
        title: 'Feature Space Transformations',
        icon: <TrendingUp className="w-6 h-6" />,
        description: 'Transform feature space using learned transformation matrices',
        example: 'Learn optimal rotation/scale to align features for better classification',
        code: `# Learnable transformation matrix
class FeatureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Learn rotation and scaling
        self.rotation = nn.Parameter(torch.eye(2))
        self.scale = nn.Parameter(torch.ones(2))
    
    def forward(self, x):
        # Apply learned transformation
        transformed = x @ self.rotation.T * self.scale
        return transformed`,
        realWorld: 'Domain adaptation, transfer learning, feature alignment'
      },
      {
        title: 'Geometric Deep Learning',
        icon: <Zap className="w-6 h-6" />,
        description: 'Apply geometric transformations that preserve structure',
        example: 'Graph neural networks use rotation-invariant transformations',
        code: `# Rotation-invariant graph convolution
# Transform node features using rotation matrices
# Preserve geometric structure while learning

def geometric_conv(node_features, rotation_matrix):
    # Rotate features, then apply convolution
    rotated_features = node_features @ rotation_matrix.T
    return graph_conv(rotated_features)`,
        realWorld: 'Molecular property prediction, 3D point clouds, geometric data analysis'
      }
    ]
  };

  const currentUseCases = useCases[operationType] || useCases['matrix-multiplication'];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-2xl font-bold text-gray-900 mb-6">
        Real-World ML Applications
      </h3>
      
      <div className="space-y-6">
        {currentUseCases.map((useCase, idx) => (
          <div
            key={idx}
            className="border-2 border-gray-200 rounded-lg p-5 hover:border-indigo-300 transition-colors"
          >
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center text-indigo-600">
                {useCase.icon}
              </div>
              <div className="flex-1">
                <h4 className="text-lg font-bold text-gray-900 mb-2">
                  {useCase.title}
                </h4>
                <p className="text-gray-700 mb-3">
                  {useCase.description}
                </p>
                <div className="bg-indigo-50 rounded-lg p-3 mb-3">
                  <p className="text-sm font-semibold text-indigo-900 mb-1">Example:</p>
                  <p className="text-sm text-indigo-800 font-mono">
                    {useCase.example}
                  </p>
                </div>
                <div className="bg-gray-900 rounded-lg p-3 mb-3">
                  <p className="text-xs text-gray-400 mb-2">Code Example:</p>
                  <pre className="text-xs text-green-400 overflow-x-auto">
                    {useCase.code}
                  </pre>
                </div>
                <div className="bg-green-50 rounded-lg p-3">
                  <p className="text-xs font-semibold text-green-900 mb-1">Real-World Usage:</p>
                  <p className="text-xs text-green-800">
                    {useCase.realWorld}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

