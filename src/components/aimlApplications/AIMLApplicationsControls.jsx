import React from 'react';

export default function AIMLApplicationsControls({ 
  selectedApplication, 
  setSelectedApplication
}) {
  const applications = [
    { id: 'image-classification', label: 'Image Classification', icon: '🖼️' },
    { id: 'sentiment-analysis', label: 'Sentiment Analysis (NLP)', icon: '💬' },
    { id: 'recommendation-system', label: 'Recommendation System', icon: '⭐' },
    { id: 'time-series-forecasting', label: 'Time Series Forecasting', icon: '📈' },
    { id: 'object-detection', label: 'Object Detection', icon: '🎯' },
    { id: 'text-generation', label: 'Text Generation', icon: '✍️' },
    { id: 'anomaly-detection', label: 'Anomaly Detection', icon: '🔍' },
    { id: 'image-generation', label: 'Image Generation', icon: '🎨' },
    { id: 'speech-recognition', label: 'Speech Recognition', icon: '🎤' },
    { id: 'machine-translation', label: 'Machine Translation', icon: '🌐' },
    { id: 'image-segmentation', label: 'Image Segmentation', icon: '✂️' },
    { id: 'reinforcement-learning', label: 'Reinforcement Learning', icon: '🎮' },
    { id: 'transfer-learning', label: 'Transfer Learning', icon: '🔄' },
    { id: 'pretrained-models', label: 'Pre-trained Models', icon: '📦' },
    { id: 'nba-chatbot', label: 'NBA Basketball Chatbot (Complete Tutorial)', icon: '🏀' },
    { id: 'maze-solver', label: 'Maze Solver (Pathfinding)', icon: '🧩' },
    { id: 'gradient-descent', label: 'Gradient Descent Visualization', icon: '📉' },
    { id: 'neural-network-playground', label: 'Neural Network Playground', icon: '🧠' },
    { id: 'linear-regression', label: 'Linear Regression Visualization', icon: '📊' },
    { id: 'pca-visualization', label: 'PCA Visualization', icon: '🔍' },
    { id: 'convolution-visualization', label: 'Convolution Operation Visualization', icon: '🔄' },
    { id: 'k-means-clustering', label: 'K-Means Clustering Visualization', icon: '🎯' },
    { id: 'trading-tools', label: 'AI Trading Tools (Complete Tutorial)', icon: '📊' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Applications</h2>

      {/* Application Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Application
        </label>
        <select
          value={selectedApplication}
          onChange={(e) => setSelectedApplication(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        >
          {applications.map(app => (
            <option key={app.id} value={app.id}>
              {app.icon} {app.label}
            </option>
          ))}
        </select>
      </div>

      {/* Info Panel */}
      <div className="bg-purple-50 rounded-lg p-4">
        <h3 className="font-semibold text-purple-900 mb-2">About This Application</h3>
        <p className="text-sm text-purple-800">
          {selectedApplication === 'image-classification' && 
            'Build a CNN to classify images. Uses Linear Algebra (convolution operations), Calculus (backpropagation), and Neural Networks.'}
          {selectedApplication === 'sentiment-analysis' && 
            'Analyze text sentiment using NLP. Combines Probability (word distributions), Neural Networks (RNN/LSTM), and Supervised Learning.'}
          {selectedApplication === 'recommendation-system' && 
            'Build a recommendation system using collaborative filtering. Uses Linear Algebra (matrix factorization) and Unsupervised Learning.'}
          {selectedApplication === 'time-series-forecasting' && 
            'Forecast future values using LSTM networks. Combines Calculus (gradient descent), Neural Networks, and Time Series Analysis.'}
          {selectedApplication === 'object-detection' && 
            'Detect and locate objects in images. Uses CNNs, Linear Algebra (feature maps), and Computer Vision techniques.'}
          {selectedApplication === 'text-generation' && 
            'Generate text using language models. Combines Transformers, Probability (next-word prediction), and Neural Networks.'}
          {selectedApplication === 'anomaly-detection' && 
            'Detect anomalies in data. Uses Unsupervised Learning, Probability (statistical distributions), and Distance Metrics.'}
          {selectedApplication === 'image-generation' && 
            'Generate images using GANs or Diffusion Models. Combines Neural Networks, Probability, and Generative Models.'}
          {selectedApplication === 'speech-recognition' && 
            'Convert speech to text using audio processing. Uses CNNs/RNNs, MFCC features, and Sequence-to-Sequence models.'}
          {selectedApplication === 'machine-translation' && 
            'Translate between languages using Seq2Seq models. Combines Transformers, Attention mechanisms, and Neural Networks.'}
          {selectedApplication === 'image-segmentation' && 
            'Segment images at pixel level. Uses CNNs, U-Net architecture, and Supervised Learning for semantic/instance segmentation.'}
          {selectedApplication === 'reinforcement-learning' && 
            'Learn through interaction and rewards. Uses Q-learning, Policy Gradients, and different optimization paradigm.'}
          {selectedApplication === 'transfer-learning' && 
            'Leverage pre-trained models for new tasks. Combines Neural Networks, Fine-tuning, and Feature Extraction techniques.'}
          {selectedApplication === 'pretrained-models' && 
            'Use pre-trained models for immediate inference. Covers BERT, GPT, ResNet, and other popular models ready to use.'}
          {selectedApplication === 'nba-chatbot' && 
            'Complete step-by-step tutorial building an NBA basketball chatbot. Covers all math, concepts, theory, and formulas from scratch.'}
          {selectedApplication === 'maze-solver' && 
            'Interactive maze solver with pathfinding algorithms. Visualize A*, Dijkstra, BFS, and DFS algorithms on 8x8+ grids.'}
          {selectedApplication === 'gradient-descent' && 
            'Visualize how gradient descent finds the minimum of loss functions. Interactive demonstration of optimization, learning rates, and convergence.'}
          {selectedApplication === 'neural-network-playground' && 
            'Interactive neural network visualization. See forward pass, backpropagation, and weight updates in real-time. Understand how networks learn.'}
          {selectedApplication === 'linear-regression' && 
            'Visualize linear regression with gradient descent. See how calculus and linear algebra work together to fit a line to data points.'}
          {selectedApplication === 'pca-visualization' && 
            'Principal Component Analysis visualization. See eigenvalues, eigenvectors, and dimensionality reduction in action using Linear Algebra.'}
          {selectedApplication === 'convolution-visualization' && 
            'Interactive convolution operation demo. Understand how matrix operations work in CNNs with step-by-step visualization of the convolution process.'}
          {selectedApplication === 'k-means-clustering' && 
            'Visualize K-Means clustering algorithm. See how data points are grouped into clusters using distance metrics and centroid updates from Linear Algebra and Statistics.'}
        </p>
      </div>

      {/* Concepts Used */}
      <div className="mt-6 bg-gray-50 rounded-lg p-4">
        <h3 className="font-semibold text-gray-900 mb-2">Concepts Used</h3>
        <div className="flex flex-wrap gap-2">
          {selectedApplication === 'image-classification' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Neural Networks</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Supervised Learning</span>
            </>
          )}
          {selectedApplication === 'sentiment-analysis' && (
            <>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Probability</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Neural Networks</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Supervised Learning</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">NLP</span>
            </>
          )}
          {selectedApplication === 'recommendation-system' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-teal-100 text-teal-800 rounded text-xs">Unsupervised Learning</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Matrix Factorization</span>
            </>
          )}
          {selectedApplication === 'time-series-forecasting' && (
            <>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Neural Networks</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">LSTM</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Supervised Learning</span>
            </>
          )}
          {selectedApplication === 'object-detection' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">CNNs</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Supervised Learning</span>
            </>
          )}
          {selectedApplication === 'text-generation' && (
            <>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Probability</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Transformers</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Attention</span>
            </>
          )}
          {selectedApplication === 'anomaly-detection' && (
            <>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Probability</span>
              <span className="px-2 py-1 bg-teal-100 text-teal-800 rounded text-xs">Unsupervised Learning</span>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Distance Metrics</span>
            </>
          )}
          {selectedApplication === 'image-generation' && (
            <>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Probability</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">GANs/Diffusion</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Generative Models</span>
            </>
          )}
          {selectedApplication === 'speech-recognition' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Signal Processing</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">RNNs/CNNs</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Sequence Models</span>
            </>
          )}
          {selectedApplication === 'machine-translation' && (
            <>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Transformers</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Attention</span>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Seq2Seq</span>
            </>
          )}
          {selectedApplication === 'image-segmentation' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">CNNs</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">U-Net</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Supervised Learning</span>
            </>
          )}
          {selectedApplication === 'reinforcement-learning' && (
            <>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Q-Learning</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Policy Gradients</span>
              <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded text-xs">RL</span>
            </>
          )}
          {selectedApplication === 'transfer-learning' && (
            <>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Pre-trained Models</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Fine-tuning</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Feature Extraction</span>
            </>
          )}
          {selectedApplication === 'pretrained-models' && (
            <>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">BERT/GPT</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">ResNet/EfficientNet</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Inference</span>
            </>
          )}
          {selectedApplication === 'nba-chatbot' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Probability</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">NLP</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Complete Tutorial</span>
            </>
          )}
          {selectedApplication === 'maze-solver' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Graph Theory</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">A* Algorithm</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Heuristics</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Pathfinding</span>
            </>
          )}
          {selectedApplication === 'gradient-descent' && (
            <>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Derivatives</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Optimization</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Learning Rate</span>
            </>
          )}
          {selectedApplication === 'neural-network-playground' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Backpropagation</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Neural Networks</span>
            </>
          )}
          {selectedApplication === 'linear-regression' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Linear Algebra</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Calculus</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Gradient Descent</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Least Squares</span>
            </>
          )}
          {selectedApplication === 'pca-visualization' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Eigenvalues</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Eigenvectors</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Covariance</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Dimensionality Reduction</span>
            </>
          )}
          {selectedApplication === 'convolution-visualization' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Matrix Operations</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Convolution</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">CNNs</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Feature Extraction</span>
            </>
          )}
          {selectedApplication === 'k-means-clustering' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Euclidean Distance</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">Centroids</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Unsupervised Learning</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Statistics</span>
            </>
          )}
          {selectedApplication === 'trading-tools' && (
            <>
              <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded text-xs">Time Series</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs">LSTM</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">Reinforcement Learning</span>
              <span className="px-2 py-1 bg-violet-100 text-violet-800 rounded text-xs">Technical Indicators</span>
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">Risk Management</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

