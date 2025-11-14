import React from 'react';

export default function AIMLApplicationsEducationalPanels({ selectedApplication }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">How Concepts Come Together</h2>

      {selectedApplication === 'image-classification' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Image Classification with CNNs</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Image Classification?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Image classification is one of the most fundamental computer vision tasks. It's used when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Categorize images into predefined classes (e.g., cat vs. dog, different types of diseases)</li>
                <li>Automate visual inspection and quality control</li>
                <li>Enable machines to "see" and understand visual content</li>
                <li>Build the foundation for more complex vision tasks (object detection, segmentation)</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Image Classification:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather labeled images for each class you want to recognize</li>
                <li><strong>Data Preprocessing:</strong> Resize images, normalize pixel values, apply augmentations</li>
                <li><strong>Model Selection:</strong> Choose CNN architecture (ResNet, VGG, EfficientNet) or build custom</li>
                <li><strong>Training:</strong> Train on labeled data using supervised learning with backpropagation</li>
                <li><strong>Evaluation:</strong> Test on held-out data, measure accuracy, precision, recall</li>
                <li><strong>Deployment:</strong> Use trained model to classify new images in production</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Convolution operations use matrix multiplication to extract features from images</li>
              <li><strong>Calculus:</strong> Backpropagation uses gradients to update weights during training</li>
              <li><strong>Neural Networks:</strong> Multi-layer architecture with convolutional, pooling, and fully connected layers</li>
              <li><strong>Supervised Learning:</strong> Trained on labeled image datasets (e.g., CIFAR-10, ImageNet)</li>
              <li><strong>Probability:</strong> Output probabilities for each class using softmax activation</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Convolution Operation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">(I * K)[i,j] = Σ<sub>m</sub> Σ<sub>n</sub> I[i+m, j+n] · K[m, n]</p>
                <p className="text-gray-600 text-xs mt-1">where I = input image, K = kernel/filter</p>
              </div>
              <div>
                <strong className="text-gray-800">Cross-Entropy Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = -Σ y_i · log(ŷ_i)</p>
                <p className="text-gray-600 text-xs mt-1">where y_i = true label, ŷ_i = predicted probability</p>
              </div>
              <div>
                <strong className="text-gray-800">Softmax Activation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(y_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <p className="text-gray-600 text-xs mt-1">Converts logits to class probabilities</p>
              </div>
              <div>
                <strong className="text-gray-800">Gradient Descent:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ<sub>{'{'}t+1{'}'}</sub> = θ<sub>t</sub> - α · ∇<sub>θ</sub> L(θ<sub>t</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Updates weights to minimize loss</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Medical image diagnosis (X-rays, MRIs, CT scans)</li>
              <li>Autonomous vehicles (road sign recognition, lane detection)</li>
              <li>Quality control in manufacturing (defect detection)</li>
              <li>Security systems (facial recognition, access control)</li>
              <li>E-commerce (product categorization, visual search)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'sentiment-analysis' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Sentiment Analysis</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Sentiment Analysis?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Sentiment analysis helps you understand how people feel about your product, service, or brand. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Monitor brand reputation across social media and reviews</li>
                <li>Understand customer satisfaction and pain points</li>
                <li>Automate customer support ticket prioritization</li>
                <li>Analyze market trends and public opinion</li>
                <li>Filter and categorize user-generated content</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Sentiment Analysis:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather text data (reviews, tweets, comments, surveys)</li>
                <li><strong>Labeling:</strong> Manually label samples as positive, negative, or neutral (or use existing datasets)</li>
                <li><strong>Text Preprocessing:</strong> Tokenize, remove stop words, handle emojis and special characters</li>
                <li><strong>Model Selection:</strong> Choose between RNN/LSTM, Transformer models, or pre-trained models (BERT)</li>
                <li><strong>Training:</strong> Train on labeled data, fine-tune pre-trained models for your domain</li>
                <li><strong>Deployment:</strong> Integrate into your system to analyze incoming text in real-time</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Probability:</strong> Word embeddings and language models use probability distributions</li>
              <li><strong>Neural Networks:</strong> RNNs/LSTMs process sequential text data</li>
              <li><strong>Supervised Learning:</strong> Trained on labeled text data</li>
              <li><strong>Linear Algebra:</strong> Word embeddings are vectors in high-dimensional space</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Word Embeddings:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">e_i = E[i] where E ∈ R^(V×d)</p>
                <p className="text-gray-600 text-xs mt-1">where V = vocabulary size, d = embedding dimension</p>
              </div>
              <div>
                <strong className="text-gray-800">LSTM Cell:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">h<sub>t</sub> = LSTM(x<sub>t</sub>, h<sub>{'{'}t-1{'}'}</sub>, c<sub>{'{'}t-1{'}'}</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Processes sequential text with memory</p>
              </div>
              <div>
                <strong className="text-gray-800">Cross-Entropy Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = -Σ y_i · log(ŷ_i)</p>
                <p className="text-gray-600 text-xs mt-1">Measures prediction error</p>
              </div>
              <div>
                <strong className="text-gray-800">Softmax:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(y_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <p className="text-gray-600 text-xs mt-1">Outputs sentiment probabilities</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Social media monitoring (Twitter, Facebook, Instagram)</li>
              <li>Customer feedback analysis (reviews, support tickets)</li>
              <li>Brand reputation management</li>
              <li>Market research and competitive analysis</li>
              <li>News article sentiment tracking</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'recommendation-system' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Recommendation System</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Recommendation Systems?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Recommendation systems help users discover relevant content and increase engagement. Use them when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Increase user engagement and time spent on your platform</li>
                <li>Boost sales by suggesting relevant products</li>
                <li>Handle large catalogs where users can't browse everything</li>
                <li>Personalize user experience based on preferences</li>
                <li>Reduce information overload by filtering content</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Recommendation Systems:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather user-item interactions (ratings, purchases, views, clicks)</li>
                <li><strong>Choose Approach:</strong> Collaborative filtering (user-based or item-based) or content-based</li>
                <li><strong>Matrix Factorization:</strong> Decompose user-item matrix into lower-dimensional embeddings</li>
                <li><strong>Training:</strong> Learn latent factors that capture user preferences and item characteristics</li>
                <li><strong>Prediction:</strong> Predict ratings for unseen user-item pairs</li>
                <li><strong>Ranking:</strong> Sort items by predicted rating and recommend top-K items</li>
                <li><strong>Evaluation:</strong> Measure using precision@K, recall@K, or NDCG metrics</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Matrix factorization decomposes user-item interaction matrix</li>
              <li><strong>Unsupervised Learning:</strong> Collaborative filtering finds patterns without explicit labels</li>
              <li><strong>Probability:</strong> Bayesian approaches for recommendation</li>
              <li><strong>Distance Metrics:</strong> Similarity measures between users/items</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Matrix Factorization:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">R ≈ U · V^T</p>
                <p className="text-gray-600 text-xs mt-1">where R = user-item matrix, U = user embeddings, V = item embeddings</p>
              </div>
              <div>
                <strong className="text-gray-800">Predicted Rating:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">ŷ<sub>u,i</sub> = u<sub>u</sub><sup>T</sup> · v<sub>i</sub></p>
                <p className="text-gray-600 text-xs mt-1">where u<sub>u</sub> = user vector, v<sub>i</sub> = item vector</p>
              </div>
              <div>
                <strong className="text-gray-800">Cosine Similarity:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">sim(u, v) = (u · v) / (||u|| × ||v||)</p>
                <p className="text-gray-600 text-xs mt-1">Measures similarity between users/items</p>
              </div>
              <div>
                <strong className="text-gray-800">Mean Squared Error:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">MSE = (1/n) Σ(ŷ<sub>u,i</sub> - r<sub>u,i</sub>)²</p>
                <p className="text-gray-600 text-xs mt-1">Loss function for training</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>E-commerce product recommendations (Amazon, eBay)</li>
              <li>Streaming service content suggestions (Netflix, Spotify)</li>
              <li>Music and playlist recommendations</li>
              <li>News article personalization</li>
              <li>Job recommendations on LinkedIn</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'time-series-forecasting' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Time Series Forecasting</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Time Series Forecasting?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Time series forecasting helps predict future trends and make data-driven decisions. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Plan inventory and resource allocation based on future demand</li>
                <li>Make financial decisions (investments, budgeting)</li>
                <li>Optimize operations (energy consumption, traffic flow)</li>
                <li>Detect anomalies by comparing predictions to actual values</li>
                <li>Understand trends and seasonal patterns in your data</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Time Series Forecasting:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather historical time series data (daily, hourly, etc.)</li>
                <li><strong>Exploratory Analysis:</strong> Identify trends, seasonality, and patterns</li>
                <li><strong>Preprocessing:</strong> Handle missing values, normalize data, create sequences</li>
                <li><strong>Model Selection:</strong> Choose LSTM, GRU, or Transformer models for complex patterns</li>
                <li><strong>Training:</strong> Train on historical sequences to predict next values</li>
                <li><strong>Validation:</strong> Use walk-forward validation to test on future data</li>
                <li><strong>Forecasting:</strong> Generate predictions for future time steps</li>
                <li><strong>Monitoring:</strong> Continuously update model as new data arrives</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Calculus:</strong> Gradient descent optimizes LSTM weights</li>
              <li><strong>Neural Networks:</strong> LSTM architecture handles sequential dependencies</li>
              <li><strong>Probability:</strong> Uncertainty quantification in predictions</li>
              <li><strong>Supervised Learning:</strong> Learn patterns from historical data</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">LSTM Cell:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">h<sub>t</sub> = LSTM(x<sub>t</sub>, h<sub>{'{'}t-1{'}'}</sub>, c<sub>{'{'}t-1{'}'}</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Processes time series sequences</p>
              </div>
              <div>
                <strong className="text-gray-800">Prediction:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">ŷ<sub>{'{'}t+1{'}'}</sub> = f(h<sub>t</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Forecasts next time step</p>
              </div>
              <div>
                <strong className="text-gray-800">Mean Squared Error:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">MSE = (1/T) Σ(ŷ<sub>t</sub> - y<sub>t</sub>)²</p>
                <p className="text-gray-600 text-xs mt-1">Loss function for time series</p>
              </div>
              <div>
                <strong className="text-gray-800">Gradient Descent:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ<sub>{'{'}t+1{'}'}</sub> = θ<sub>t</sub> - α · ∇<sub>θ</sub> L(θ<sub>t</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Optimizes model parameters</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Stock price and financial market prediction</li>
              <li>Weather forecasting and climate modeling</li>
              <li>Sales and demand forecasting</li>
              <li>Energy demand prediction for power grids</li>
              <li>Web traffic and user behavior prediction</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'object-detection' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Object Detection</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Object Detection?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Object detection goes beyond classification by locating objects in images. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Locate and identify multiple objects in a single image</li>
                <li>Track objects across video frames</li>
                <li>Enable robots and autonomous systems to navigate and interact</li>
                <li>Count objects or measure distances between them</li>
                <li>Understand spatial relationships in images</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Object Detection:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather images with labeled bounding boxes (x, y, width, height) and class labels</li>
                <li><strong>Dataset Preparation:</strong> Use datasets like COCO, Pascal VOC, or create custom annotations</li>
                <li><strong>Model Selection:</strong> Choose YOLO, R-CNN, SSD, or RetinaNet architectures</li>
                <li><strong>Training:</strong> Train on labeled bounding boxes using localization + classification loss</li>
                <li><strong>Post-processing:</strong> Apply Non-Maximum Suppression (NMS) to remove duplicate detections</li>
                <li><strong>Evaluation:</strong> Measure using mAP (mean Average Precision) metric</li>
                <li><strong>Deployment:</strong> Use for real-time detection in video streams or batch processing</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Feature extraction using convolutional operations</li>
              <li><strong>Neural Networks:</strong> CNNs for feature extraction, RPN for region proposals</li>
              <li><strong>Supervised Learning:</strong> Trained on labeled bounding boxes</li>
              <li><strong>Probability:</strong> Class probabilities and confidence scores</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Bounding Box Prediction:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">[x, y, w, h] = f(features)</p>
                <p className="text-gray-600 text-xs mt-1">Predicts object location and size</p>
              </div>
              <div>
                <strong className="text-gray-800">IoU (Intersection over Union):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">IoU = Area(Intersection) / Area(Union)</p>
                <p className="text-gray-600 text-xs mt-1">Measures detection accuracy</p>
              </div>
              <div>
                <strong className="text-gray-800">Localization Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L_loc = Σ smooth_L1(pred_box - true_box)</p>
                <p className="text-gray-600 text-xs mt-1">Penalizes incorrect bounding boxes</p>
              </div>
              <div>
                <strong className="text-gray-800">Classification Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L_cls = -log(P(class))</p>
                <p className="text-gray-600 text-xs mt-1">Cross-entropy for object class</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Autonomous vehicles (pedestrian detection, traffic signs)</li>
              <li>Surveillance and security systems</li>
              <li>Retail analytics (customer counting, product placement)</li>
              <li>Medical imaging (tumor detection, organ localization)</li>
              <li>Sports analytics (player tracking, ball tracking)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'text-generation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Text Generation</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Text Generation?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Text generation enables machines to create human-like content. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Create conversational AI assistants and chatbots</li>
                <li>Generate content automatically (articles, summaries, product descriptions)</li>
                <li>Assist in creative writing and storytelling</li>
                <li>Complete code or text based on partial input</li>
                <li>Translate between languages or rewrite text in different styles</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Text Generation:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather large text corpus (books, articles, code, conversations)</li>
                <li><strong>Preprocessing:</strong> Tokenize text, build vocabulary, create sequences</li>
                <li><strong>Model Selection:</strong> Use GPT-style models, LSTM, or fine-tune pre-trained models (GPT, T5)</li>
                <li><strong>Training:</strong> Train on next-word prediction task (autoregressive language modeling)</li>
                <li><strong>Sampling:</strong> Use temperature and top-k/top-p sampling for diverse outputs</li>
                <li><strong>Fine-tuning:</strong> Adapt pre-trained models to your specific domain or task</li>
                <li><strong>Deployment:</strong> Integrate into applications for real-time text generation</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Probability:</strong> Next-word prediction using probability distributions</li>
              <li><strong>Neural Networks:</strong> Transformer architecture with attention mechanisms</li>
              <li><strong>Linear Algebra:</strong> Attention matrices and embeddings</li>
              <li><strong>Calculus:</strong> Training via backpropagation</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Next-Word Probability:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(w<sub>t</sub> | w<sub>{'{'}t-1{'}'}</sub>, ..., w<sub>1</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Probability of next word given context</p>
              </div>
              <div>
                <strong className="text-gray-800">Attention:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Attention(Q, K, V) = softmax(QK^T / √d_k) · V</p>
                <p className="text-gray-600 text-xs mt-1">Transformer attention mechanism</p>
              </div>
              <div>
                <strong className="text-gray-800">Softmax:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(w_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <p className="text-gray-600 text-xs mt-1">Converts logits to word probabilities</p>
              </div>
              <div>
                <strong className="text-gray-800">Cross-Entropy Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = -Σ log(P(w_true))</p>
                <p className="text-gray-600 text-xs mt-1">Language modeling loss</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Chatbots and virtual assistants (ChatGPT, Siri, Alexa)</li>
              <li>Content creation (article writing, social media posts)</li>
              <li>Code generation (GitHub Copilot, Codex)</li>
              <li>Translation systems (Google Translate)</li>
              <li>Email and document auto-completion</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'anomaly-detection' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Anomaly Detection</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Anomaly Detection?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Anomaly detection identifies rare events that differ significantly from normal patterns. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Detect fraud, security breaches, or suspicious activities</li>
                <li>Identify equipment failures or defects before they cause problems</li>
                <li>Find rare medical conditions or unusual patient patterns</li>
                <li>Monitor system health and detect performance issues</li>
                <li>Discover data quality issues or errors in datasets</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Anomaly Detection:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather normal operation data (anomalies are rare, so you mostly have normal data)</li>
                <li><strong>Feature Engineering:</strong> Extract relevant features that capture normal vs. abnormal behavior</li>
                <li><strong>Model Selection:</strong> Choose Isolation Forest, Autoencoder, One-Class SVM, or statistical methods</li>
                <li><strong>Training:</strong> Train on normal data only (unsupervised learning)</li>
                <li><strong>Threshold Setting:</strong> Set anomaly score threshold based on false positive tolerance</li>
                <li><strong>Monitoring:</strong> Continuously score new data and flag anomalies</li>
                <li><strong>Investigation:</strong> Review flagged anomalies to understand root causes</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Probability:</strong> Statistical distributions to model normal behavior</li>
              <li><strong>Unsupervised Learning:</strong> Learn patterns without labeled anomalies</li>
              <li><strong>Distance Metrics:</strong> Measure deviation from normal patterns</li>
              <li><strong>Linear Algebra:</strong> Dimensionality reduction (PCA) for feature extraction</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Euclidean Distance:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">d(x, μ) = √(Σ(x_i - μ_i)²)</p>
                <p className="text-gray-600 text-xs mt-1">Distance from point x to centroid μ</p>
              </div>
              <div>
                <strong className="text-gray-800">Z-Score:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">z = (x - μ) / σ</p>
                <p className="text-gray-600 text-xs mt-1">Standardized deviation from mean</p>
              </div>
              <div>
                <strong className="text-gray-800">Reconstruction Error:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">E = ||x - decoder(encoder(x))||²</p>
                <p className="text-gray-600 text-xs mt-1">Autoencoder reconstruction loss</p>
              </div>
              <div>
                <strong className="text-gray-800">Isolation Score:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">score(x) = 2^(-E(h(x))/c(n))</p>
                <p className="text-gray-600 text-xs mt-1">Isolation Forest anomaly score</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Fraud detection (credit card transactions, insurance claims)</li>
              <li>Network security (intrusion detection, malware detection)</li>
              <li>Manufacturing quality control (defect detection)</li>
              <li>Medical diagnosis (rare disease detection, abnormal test results)</li>
              <li>IoT sensor monitoring (equipment failure prediction)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'image-generation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Image Generation</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Image Generation?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Image generation creates new, realistic images from scratch or based on text prompts. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Create art, illustrations, or visual content automatically</li>
                <li>Augment training datasets when real data is scarce</li>
                <li>Generate synthetic data for testing and simulation</li>
                <li>Create virtual environments and assets for games/VR</li>
                <li>Transform images (style transfer, super-resolution, inpainting)</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Image Generation:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather large dataset of images (thousands to millions)</li>
                <li><strong>Model Selection:</strong> Choose GANs (StyleGAN, BigGAN) or Diffusion Models (DALL-E, Stable Diffusion)</li>
                <li><strong>Training:</strong> Train generator to create realistic images and discriminator to distinguish real from fake</li>
                <li><strong>Adversarial Training:</strong> Generator and discriminator compete in minimax game</li>
                <li><strong>Conditional Generation:</strong> Add text or class labels to control what's generated</li>
                <li><strong>Sampling:</strong> Generate new images by sampling from learned distribution</li>
                <li><strong>Fine-tuning:</strong> Adapt pre-trained models to your specific domain or style</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Probability:</strong> Learn data distributions to generate realistic samples</li>
              <li><strong>Neural Networks:</strong> Generator and discriminator networks (GANs) or diffusion process</li>
              <li><strong>Calculus:</strong> Adversarial training optimization</li>
              <li><strong>Linear Algebra:</strong> Convolutional operations in generator networks</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">GAN Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]</p>
                <p className="text-gray-600 text-xs mt-1">Adversarial minimax game</p>
              </div>
              <div>
                <strong className="text-gray-800">Generator:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">G(z) = image</p>
                <p className="text-gray-600 text-xs mt-1">Maps noise z to image</p>
              </div>
              <div>
                <strong className="text-gray-800">Discriminator:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">D(x) = P(real | x)</p>
                <p className="text-gray-600 text-xs mt-1">Probability image is real</p>
              </div>
              <div>
                <strong className="text-gray-800">Diffusion Process:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">q(x<sub>t</sub> | x<sub>{'{'}t-1{'}'}</sub>) = N(x<sub>t</sub>; √(1-β<sub>t</sub>)x<sub>{'{'}t-1{'}'}</sub>, β<sub>t</sub> I)</p>
                <p className="text-gray-600 text-xs mt-1">Forward diffusion adds noise</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Art generation and creative content (DALL-E, Midjourney)</li>
              <li>Data augmentation for training other models</li>
              <li>Virtual world creation (game assets, environments)</li>
              <li>Style transfer and image editing</li>
              <li>Medical imaging (synthetic scans for training)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'speech-recognition' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Speech Recognition</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Speech Recognition?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Speech recognition converts spoken language into text, enabling voice-controlled applications. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Enable voice commands and voice-controlled interfaces</li>
                <li>Transcribe audio content (meetings, interviews, lectures)</li>
                <li>Build accessibility features for people with disabilities</li>
                <li>Create hands-free applications (smart speakers, car systems)</li>
                <li>Automate call center transcriptions and voice assistants</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Speech Recognition:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Audio Preprocessing:</strong> Convert audio to spectrograms, extract MFCC features</li>
                <li><strong>Feature Extraction:</strong> Use MFCC, Mel-spectrograms, or raw audio waveforms</li>
                <li><strong>Model Selection:</strong> Choose CNN+RNN, Transformer, or pre-trained models (Wav2Vec, Whisper)</li>
                <li><strong>Sequence Alignment:</strong> Use CTC loss for aligning audio sequences with text</li>
                <li><strong>Training:</strong> Train on audio-text pairs, handle variable-length sequences</li>
                <li><strong>Decoding:</strong> Use greedy decoding or beam search to convert predictions to text</li>
                <li><strong>Post-processing:</strong> Apply language models for better accuracy</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Signal Processing:</strong> MFCC features capture audio characteristics</li>
              <li><strong>Neural Networks:</strong> CNNs extract features, RNNs/LSTMs model temporal dependencies</li>
              <li><strong>Linear Algebra:</strong> Convolution operations process spectrograms</li>
              <li><strong>Calculus:</strong> Backpropagation through time for sequence models</li>
              <li><strong>Probability:</strong> CTC loss handles sequence alignment without explicit alignment</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">MFCC Features:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">MFCC = DCT(log(Mel(FFT(audio))))</p>
                <p className="text-gray-600 text-xs mt-1">Mel-frequency cepstral coefficients</p>
              </div>
              <div>
                <strong className="text-gray-800">CTC Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L_ctc = -log(Σ P(π|x)) where π ∈ B^(-1)(y)</p>
                <p className="text-gray-600 text-xs mt-1">Handles variable-length sequences</p>
              </div>
              <div>
                <strong className="text-gray-800">LSTM:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">h<sub>t</sub> = LSTM(MFCC<sub>t</sub>, h<sub>{'{'}t-1{'}'}</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Processes audio sequence</p>
              </div>
              <div>
                <strong className="text-gray-800">Softmax:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(char_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <p className="text-gray-600 text-xs mt-1">Character probabilities</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Voice assistants (Siri, Alexa, Google Assistant)</li>
              <li>Transcription services (meeting notes, podcast transcripts)</li>
              <li>Accessibility tools (voice-to-text for hearing impaired)</li>
              <li>Call center automation (voice analytics, call routing)</li>
              <li>Smart home devices (voice commands, voice search)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'machine-translation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Machine Translation</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Machine Translation?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Machine translation automatically translates text between languages. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Break down language barriers in global communication</li>
                <li>Translate content for international audiences (websites, documents)</li>
                <li>Enable real-time translation in chat applications</li>
                <li>Process multilingual data and content</li>
                <li>Support customer service across different languages</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Machine Translation:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather parallel corpora (source-target sentence pairs)</li>
                <li><strong>Preprocessing:</strong> Tokenize, normalize text, handle special characters</li>
                <li><strong>Model Selection:</strong> Use Seq2Seq models, Transformers, or pre-trained models (mBART, T5)</li>
                <li><strong>Architecture:</strong> Encoder processes source language, decoder generates target language</li>
                <li><strong>Attention Mechanism:</strong> Allows model to focus on relevant parts of source text</li>
                <li><strong>Training:</strong> Train on parallel corpora, use teacher forcing during training</li>
                <li><strong>Decoding:</strong> Use beam search for better translations</li>
                <li><strong>Evaluation:</strong> Measure using BLEU score, METEOR, or human evaluation</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Neural Networks:</strong> Encoder-decoder architecture with attention</li>
              <li><strong>Linear Algebra:</strong> Attention matrices compute relationships between words</li>
              <li><strong>Probability:</strong> Next-word prediction using probability distributions</li>
              <li><strong>Calculus:</strong> Backpropagation through encoder-decoder structure</li>
              <li><strong>Transformers:</strong> Self-attention and cross-attention mechanisms</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Attention:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Attention(Q, K, V) = softmax(QK^T / √d_k) · V</p>
                <p className="text-gray-600 text-xs mt-1">Cross-attention between source and target</p>
              </div>
              <div>
                <strong className="text-gray-800">Encoder:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">E = Encoder(source_text)</p>
                <p className="text-gray-600 text-xs mt-1">Encodes source language</p>
              </div>
              <div>
                <strong className="text-gray-800">Decoder:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">target<sub>t</sub> = Decoder(E, target<sub>{'{'}t-1{'}'}</sub>)</p>
                <p className="text-gray-600 text-xs mt-1">Generates target language</p>
              </div>
              <div>
                <strong className="text-gray-800">BLEU Score:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">BLEU = BP · exp(Σ log(p_n))</p>
                <p className="text-gray-600 text-xs mt-1">Translation quality metric</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Translation services (Google Translate, DeepL)</li>
              <li>Multilingual content management systems</li>
              <li>Real-time chat translation (Skype, WhatsApp)</li>
              <li>Document translation (legal, medical, technical documents)</li>
              <li>E-commerce (product descriptions, customer reviews)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'image-segmentation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Image Segmentation</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Image Segmentation?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Image segmentation provides pixel-level understanding of images. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Understand precise boundaries and shapes of objects</li>
                <li>Separate different objects or regions in images</li>
                <li>Enable medical image analysis (organ segmentation, tumor detection)</li>
                <li>Support autonomous vehicles (road, lane, obstacle segmentation)</li>
                <li>Create image editing tools (background removal, object isolation)</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Image Segmentation:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Data Collection:</strong> Gather images with pixel-level annotations (masks)</li>
                <li><strong>Annotation:</strong> Label each pixel with class (semantic) or instance ID (instance)</li>
                <li><strong>Model Selection:</strong> Choose U-Net, DeepLabV3, Mask R-CNN, or SegFormer</li>
                <li><strong>Architecture:</strong> Encoder-decoder with skip connections for precise boundaries</li>
                <li><strong>Training:</strong> Train on pixel-level labels, use cross-entropy or Dice loss</li>
                <li><strong>Post-processing:</strong> Apply CRF or morphological operations to refine masks</li>
                <li><strong>Evaluation:</strong> Measure using IoU (Intersection over Union), pixel accuracy, mIoU</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Neural Networks:</strong> U-Net encoder-decoder architecture with skip connections</li>
              <li><strong>Linear Algebra:</strong> Convolution operations extract spatial features</li>
              <li><strong>Calculus:</strong> Backpropagation through encoder-decoder structure</li>
              <li><strong>Supervised Learning:</strong> Trained on pixel-level labeled data</li>
              <li><strong>Probability:</strong> Pixel-wise class probabilities</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Pixel-wise Prediction:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(class_i | pixel) = softmax(features(pixel))</p>
                <p className="text-gray-600 text-xs mt-1">Probability for each pixel</p>
              </div>
              <div>
                <strong className="text-gray-800">Dice Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Dice = 2|X ∩ Y| / (|X| + |Y|)</p>
                <p className="text-gray-600 text-xs mt-1">Measures overlap between prediction and ground truth</p>
              </div>
              <div>
                <strong className="text-gray-800">IoU (Intersection over Union):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">IoU = |X ∩ Y| / |X ∪ Y|</p>
                <p className="text-gray-600 text-xs mt-1">Segmentation accuracy metric</p>
              </div>
              <div>
                <strong className="text-gray-800">Cross-Entropy Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = -Σ log(P(class_true))</p>
                <p className="text-gray-600 text-xs mt-1">Pixel-wise classification loss</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Medical imaging (organ segmentation, tumor detection, cell segmentation)</li>
              <li>Autonomous vehicles (road segmentation, lane detection, obstacle identification)</li>
              <li>Image editing (background removal, object isolation, photo manipulation)</li>
              <li>Satellite imagery (land use classification, building detection)</li>
              <li>Robotics (object manipulation, scene understanding)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'reinforcement-learning' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Reinforcement Learning</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Reinforcement Learning?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Reinforcement learning learns optimal actions through trial and error. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Solve sequential decision-making problems</li>
                <li>Train agents to play games or solve puzzles</li>
                <li>Optimize resource allocation and scheduling</li>
                <li>Control robots and autonomous systems</li>
                <li>Optimize strategies in dynamic environments</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Reinforcement Learning:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Define Environment:</strong> Create or use environment (game, simulation, real-world)</li>
                <li><strong>Define State Space:</strong> What information the agent observes</li>
                <li><strong>Define Action Space:</strong> What actions the agent can take</li>
                <li><strong>Define Reward Function:</strong> How to reward/punish agent's actions</li>
                <li><strong>Choose Algorithm:</strong> Q-learning, Policy Gradient, Actor-Critic, or PPO</li>
                <li><strong>Training:</strong> Agent interacts with environment, learns from rewards</li>
                <li><strong>Exploration:</strong> Balance exploration (try new actions) vs exploitation (use best actions)</li>
                <li><strong>Evaluation:</strong> Test trained agent, measure cumulative rewards</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Markov Decision Process:</strong> Mathematical framework for RL problems</li>
              <li><strong>Q-Learning:</strong> Learn action-value function Q(s,a)</li>
              <li><strong>Policy Gradient:</strong> Directly optimize policy using gradients</li>
              <li><strong>Calculus:</strong> Gradient descent for policy/value function optimization</li>
              <li><strong>Probability:</strong> Policy as probability distribution over actions</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Q-Learning Update:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]</p>
                <p className="text-gray-600 text-xs mt-1">where α = learning rate, γ = discount factor</p>
              </div>
              <div>
                <strong className="text-gray-800">Policy Gradient:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">∇J(θ) = E[∇log π(a|s) · Q(s,a)]</p>
                <p className="text-gray-600 text-xs mt-1">Gradient of policy objective</p>
              </div>
              <div>
                <strong className="text-gray-800">Value Function:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">V(s) = E[Σ γ^t r_t | s_0 = s]</p>
                <p className="text-gray-600 text-xs mt-1">Expected cumulative reward</p>
              </div>
              <div>
                <strong className="text-gray-800">Bellman Equation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">V(s) = max_a [r(s,a) + γ Σ P(s'|s,a) V(s')]</p>
                <p className="text-gray-600 text-xs mt-1">Optimal value function</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Game playing (AlphaGo, OpenAI Five, Atari games)</li>
              <li>Robotics (robot control, manipulation, navigation)</li>
              <li>Autonomous vehicles (driving policies, traffic optimization)</li>
              <li>Recommendation systems (dynamic recommendations, A/B testing)</li>
              <li>Resource management (data center optimization, energy management)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'transfer-learning' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Transfer Learning</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Transfer Learning?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Transfer learning leverages knowledge from pre-trained models. Use it when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Train models with limited data (common in real-world scenarios)</li>
                <li>Reduce training time and computational resources</li>
                <li>Improve performance by leveraging features learned on large datasets</li>
                <li>Adapt models to new domains or tasks quickly</li>
                <li>Build production-ready models faster</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Transfer Learning:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Choose Pre-trained Model:</strong> Select model trained on similar data (ImageNet for images, BERT for text)</li>
                <li><strong>Feature Extraction:</strong> Freeze pre-trained layers, use as feature extractor</li>
                <li><strong>Add Custom Head:</strong> Replace final layers with task-specific classifier</li>
                <li><strong>Fine-tuning:</strong> Unfreeze some layers, train with lower learning rate</li>
                <li><strong>Progressive Unfreezing:</strong> Gradually unfreeze layers during training</li>
                <li><strong>Domain Adaptation:</strong> Adapt to different data distribution if needed</li>
                <li><strong>Evaluation:</strong> Compare performance with/without transfer learning</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Neural Networks:</strong> Reuse learned feature representations</li>
              <li><strong>Supervised Learning:</strong> Fine-tune on new labeled data</li>
              <li><strong>Calculus:</strong> Gradient descent with different learning rates for different layers</li>
              <li><strong>Feature Learning:</strong> Lower layers learn general features, higher layers learn task-specific</li>
              <li><strong>Domain Adaptation:</strong> Adapt features to new data distributions</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">Feature Extraction:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">features = f_base(x)</p>
                <p className="text-gray-600 text-xs mt-1">Extract features using frozen base model</p>
              </div>
              <div>
                <strong className="text-gray-800">Fine-tuning:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ_new = θ_old - α_fine · ∇L(θ_old)</p>
                <p className="text-gray-600 text-xs mt-1">Lower learning rate α_fine &lt; α_base</p>
              </div>
              <div>
                <strong className="text-gray-800">Loss Function:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = L_task + λ · L_regularization</p>
                <p className="text-gray-600 text-xs mt-1">Task loss + regularization</p>
              </div>
              <div>
                <strong className="text-gray-800">Domain Adaptation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = L_task + λ · L_domain</p>
                <p className="text-gray-600 text-xs mt-1">Minimize domain discrepancy</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Medical imaging (transfer from ImageNet to medical images)</li>
              <li>Custom image classification (fine-tune ResNet/EfficientNet for specific classes)</li>
              <li>NLP tasks (fine-tune BERT/GPT for sentiment, classification, QA)</li>
              <li>Domain-specific models (transfer from general to specialized domains)</li>
              <li>Production ML systems (quickly adapt models to new use cases)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'pretrained-models' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Pre-trained Models</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Use Pre-trained Models?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Pre-trained models are ready-to-use models trained on large datasets. Use them when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Get immediate results without training from scratch</li>
                <li>Save time and computational resources</li>
                <li>Leverage models trained on massive datasets (ImageNet, Wikipedia, Common Crawl)</li>
                <li>Build prototypes and proof-of-concepts quickly</li>
                <li>Use state-of-the-art models without deep ML expertise</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How to Use Pre-trained Models:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Choose Model:</strong> Select appropriate pre-trained model for your task (BERT for NLP, ResNet for vision)</li>
                <li><strong>Install Libraries:</strong> Install transformers (Hugging Face) or torchvision for vision models</li>
                <li><strong>Load Model:</strong> Use model loading functions (from_pretrained, models.resnet50)</li>
                <li><strong>Preprocess Input:</strong> Format your data according to model requirements (tokenization, image preprocessing)</li>
                <li><strong>Run Inference:</strong> Pass preprocessed input to model, get predictions</li>
                <li><strong>Post-process:</strong> Decode outputs (convert token IDs to text, apply softmax for probabilities)</li>
                <li><strong>Use Pipelines:</strong> For easiest usage, use Hugging Face pipelines</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Popular Pre-trained Models:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>NLP Models:</strong> BERT, GPT-2, GPT-3, T5, RoBERTa, DistilBERT</li>
              <li><strong>Vision Models:</strong> ResNet, EfficientNet, VGG, MobileNet, YOLO, CLIP</li>
              <li><strong>Multimodal:</strong> CLIP (image-text), DALL-E, Stable Diffusion</li>
              <li><strong>Speech:</strong> Whisper, Wav2Vec, SpeechT5</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Neural Networks:</strong> Pre-trained models use various architectures (Transformers, CNNs)</li>
              <li><strong>Transfer Learning:</strong> Models trained on one task can be adapted to others</li>
              <li><strong>Feature Extraction:</strong> Use pre-trained models as feature extractors</li>
              <li><strong>Inference:</strong> Forward pass through trained model for predictions</li>
              <li><strong>Model Hubs:</strong> Platforms like Hugging Face provide thousands of models</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-3">
              <div>
                <strong className="text-gray-800">BERT Embeddings:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">E = [token_emb + position_emb + segment_emb]</p>
                <p className="text-gray-600 text-xs mt-1">Combined input embeddings</p>
              </div>
              <div>
                <strong className="text-gray-800">Attention (Transformer):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Attention(Q, K, V) = softmax(QK^T / √d_k) · V</p>
                <p className="text-gray-600 text-xs mt-1">Self-attention mechanism</p>
              </div>
              <div>
                <strong className="text-gray-800">ResNet Forward Pass:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">y = F(x) + x</p>
                <p className="text-gray-600 text-xs mt-1">Residual connection</p>
              </div>
              <div>
                <strong className="text-gray-800">Softmax Output:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(class_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <p className="text-gray-600 text-xs mt-1">Class probabilities</p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Chatbots and virtual assistants (GPT, BERT)</li>
              <li>Image classification APIs (ResNet, EfficientNet)</li>
              <li>Object detection systems (YOLO, Faster R-CNN)</li>
              <li>Translation services (mBART, T5)</li>
              <li>Content moderation (BERT for text, vision models for images)</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'nba-chatbot' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">NBA Basketball Chatbot - Complete Tutorial</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Build This Chatbot?</h4>
              <p className="text-blue-800 text-sm mb-2">
                This comprehensive tutorial demonstrates how all mathematical foundations work together in a real application:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>See Linear Algebra, Calculus, and Probability in action</li>
                <li>Understand how theory translates to code</li>
                <li>Learn formulas and their practical applications</li>
                <li>Build a complete working system from scratch</li>
                <li>Apply all concepts learned throughout the tutorial</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">Tutorial Structure:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Step 1 - Data Collection:</strong> Gather NBA Q&A data, preprocessing, vocabulary building</li>
                <li><strong>Step 2 - Mathematical Foundations:</strong> Word embeddings, positional encoding, cosine similarity</li>
                <li><strong>Step 3 - Neural Architecture:</strong> LSTM/Transformer models, attention mechanisms</li>
                <li><strong>Step 4 - Training:</strong> Loss functions, gradient descent, backpropagation</li>
                <li><strong>Step 5 - Inference:</strong> Probability distributions, sampling strategies</li>
                <li><strong>Step 6 - Complete System:</strong> Full implementation integrating all components</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas Covered:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Word Embeddings */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Word Embeddings:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">e_i = E[i] where E ∈ R^(V×d)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>e_i</strong>: Embedding vector for word i (output)</li>
                    <li><strong>E</strong>: Embedding matrix of size V×d</li>
                    <li><strong>V</strong>: Vocabulary size (number of unique words)</li>
                    <li><strong>d</strong>: Embedding dimension (e.g., 128, 256, 512)</li>
                    <li><strong>E[i]</strong>: Row i of matrix E (the embedding for word i)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Vocabulary size V = 10,000 words</li>
                    <li>Embedding dimension d = 128</li>
                    <li>E is a 10,000 × 128 matrix</li>
                    <li>For word "basketball" with index i = 42:</li>
                    <li className="ml-4">e_42 = E[42] = [0.23, -0.15, 0.67, ..., 0.31] (128-dimensional vector)</li>
                    <li>Each word gets a unique 128-dimensional vector representation</li>
                  </ul>
                </div>
              </div>

              {/* Positional Encoding */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Positional Encoding:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">PE(pos, 2i) = sin(pos / 10000^(2i/d_model))</p>
                <p className="text-gray-700 text-sm font-mono">PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>PE(pos, 2i)</strong>: Positional encoding at position pos, even dimension 2i</li>
                    <li><strong>PE(pos, 2i+1)</strong>: Positional encoding at position pos, odd dimension 2i+1</li>
                    <li><strong>pos</strong>: Position in sequence (0, 1, 2, ...)</li>
                    <li><strong>i</strong>: Dimension index (0, 1, 2, ..., d_model/2 - 1)</li>
                    <li><strong>d_model</strong>: Model dimension (e.g., 256, 512)</li>
                    <li><strong>10000</strong>: Base constant for frequency scaling</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Position pos = 3 (4th word in sequence)</li>
                    <li>d_model = 256</li>
                    <li>For dimension i = 0 (first pair):</li>
                    <li className="ml-4">PE(3, 0) = sin(3 / 10000^(0/256)) = sin(3 / 10000^0) = sin(3 / 1) = sin(3) ≈ 0.141</li>
                    <li className="ml-4">PE(3, 1) = cos(3 / 10000^0) = cos(3) ≈ -0.990</li>
                    <li>For dimension i = 1 (second pair):</li>
                    <li className="ml-4">PE(3, 2) = sin(3 / 10000^(2/256)) = sin(3 / 10000^0.0078) ≈ sin(3 / 1.018) ≈ sin(2.947) ≈ 0.199</li>
                    <li className="ml-4">PE(3, 3) = cos(3 / 10000^0.0078) ≈ cos(2.947) ≈ -0.980</li>
                    <li>Each position gets a unique 256-dimensional encoding vector</li>
                  </ul>
                </div>
              </div>

              {/* Attention */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">3. Attention:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Attention(Q, K, V) = softmax(QK^T / √d_k) · V</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Q</strong>: Query matrix (what we're looking for)</li>
                    <li><strong>K</strong>: Key matrix (what we're matching against)</li>
                    <li><strong>V</strong>: Value matrix (the actual information)</li>
                    <li><strong>QK^T</strong>: Matrix multiplication of Q and transpose of K</li>
                    <li><strong>√d_k</strong>: Square root of key dimension (scaling factor)</li>
                    <li><strong>softmax</strong>: Normalizes scores to probabilities</li>
                    <li><strong>·</strong>: Matrix multiplication</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Sequence length = 4, d_k = 64</li>
                    <li>Q, K, V are 4 × 64 matrices</li>
                    <li><strong>Step 1:</strong> Compute QK^T (4 × 64) × (64 × 4) = 4 × 4 attention scores</li>
                    <li className="ml-4">Example: QK^T[0,1] = 12.5 (how much word 0 attends to word 1)</li>
                    <li><strong>Step 2:</strong> Scale by √d_k = √64 = 8</li>
                    <li className="ml-4">Scaled scores: QK^T / 8 = 12.5 / 8 = 1.5625</li>
                    <li><strong>Step 3:</strong> Apply softmax to get attention weights (probabilities)</li>
                    <li className="ml-4">softmax([1.5625, 0.8, -0.3, 0.1]) ≈ [0.45, 0.25, 0.15, 0.15]</li>
                    <li><strong>Step 4:</strong> Multiply by V: (4 × 4) × (4 × 64) = 4 × 64 output</li>
                    <li>Result: Weighted combination of values based on attention scores</li>
                  </ul>
                </div>
              </div>

              {/* Cross-Entropy Loss */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">4. Cross-Entropy Loss:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = -Σ y_i · log(ŷ_i)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>L</strong>: Total loss value (scalar)</li>
                    <li><strong>y_i</strong>: True label (1 for correct class, 0 for others)</li>
                    <li><strong>ŷ_i</strong>: Predicted probability for class i</li>
                    <li><strong>log</strong>: Natural logarithm</li>
                    <li><strong>Σ</strong>: Sum over all classes</li>
                    <li><strong>-</strong>: Negative sign (because we minimize loss)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>3 classes: ["basketball", "player", "team"]</li>
                    <li>True label: "basketball" (index 0)</li>
                    <li>True distribution: y = [1, 0, 0]</li>
                    <li>Predicted probabilities: ŷ = [0.7, 0.2, 0.1]</li>
                    <li><strong>Calculation:</strong></li>
                    <li className="ml-4">L = -(y_0·log(ŷ_0) + y_1·log(ŷ_1) + y_2·log(ŷ_2))</li>
                    <li className="ml-4">L = -(1·log(0.7) + 0·log(0.2) + 0·log(0.1))</li>
                    <li className="ml-4">L = -(log(0.7) + 0 + 0)</li>
                    <li className="ml-4">L = -(-0.357) = 0.357</li>
                    <li>If prediction was perfect: ŷ = [1.0, 0.0, 0.0], then L = -log(1.0) = 0</li>
                    <li>If prediction was wrong: ŷ = [0.1, 0.8, 0.1], then L = -log(0.1) = 2.303 (higher loss)</li>
                  </ul>
                </div>
              </div>

              {/* Gradient Descent */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">5. Gradient Descent:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ<sub>{'{'}t+1{'}'}</sub> = θ<sub>t</sub> - α · ∇<sub>θ</sub> L(θ<sub>t</sub>)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>θ<sub>{'{'}t+1{'}'}</sub></strong>: Updated parameters at step t+1</li>
                    <li><strong>θ<sub>t</sub></strong>: Current parameters at step t</li>
                    <li><strong>α</strong>: Learning rate (step size, e.g., 0.001, 0.01)</li>
                    <li><strong>∇<sub>θ</sub> L(θ<sub>t</sub>)</strong>: Gradient (derivative) of loss with respect to parameters</li>
                    <li><strong>-</strong>: Move in opposite direction of gradient (downhill)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Initial weight: θ_0 = 0.5</li>
                    <li>Learning rate: α = 0.01</li>
                    <li>Loss function: L(θ) = (θ - 1)^2 (minimum at θ = 1)</li>
                    <li><strong>Step 1:</strong> Calculate gradient</li>
                    <li className="ml-4">∇L = dL/dθ = 2(θ - 1) = 2(0.5 - 1) = -1.0</li>
                    <li><strong>Step 2:</strong> Update parameter</li>
                    <li className="ml-4">θ_1 = θ_0 - α · ∇L = 0.5 - 0.01 · (-1.0) = 0.5 + 0.01 = 0.51</li>
                    <li><strong>Step 3:</strong> Repeat</li>
                    <li className="ml-4">∇L at θ_1 = 2(0.51 - 1) = -0.98</li>
                    <li className="ml-4">θ_2 = 0.51 - 0.01 · (-0.98) = 0.51 + 0.0098 = 0.5198</li>
                    <li>After many steps, θ converges to 1.0 (the minimum)</li>
                    <li>If α too large (e.g., 0.5): might overshoot and diverge</li>
                    <li>If α too small (e.g., 0.0001): converges very slowly</li>
                  </ul>
                </div>
              </div>

              {/* Softmax */}
              <div>
                <strong className="text-gray-800 text-base">6. Softmax:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">P(y_i) = exp(z_i) / Σ_j exp(z_j)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>P(y_i)</strong>: Probability of class i (output)</li>
                    <li><strong>z_i</strong>: Logit (raw score) for class i</li>
                    <li><strong>exp</strong>: Exponential function (e^x)</li>
                    <li><strong>Σ_j exp(z_j)</strong>: Sum of exponentials of all logits (normalization constant)</li>
                    <li><strong>j</strong>: Index over all classes</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>3 classes with logits: z = [2.0, 1.0, 0.1]</li>
                    <li><strong>Step 1:</strong> Compute exponentials</li>
                    <li className="ml-4">exp(2.0) = 7.389</li>
                    <li className="ml-4">exp(1.0) = 2.718</li>
                    <li className="ml-4">exp(0.1) = 1.105</li>
                    <li><strong>Step 2:</strong> Compute sum</li>
                    <li className="ml-4">Σ_j exp(z_j) = 7.389 + 2.718 + 1.105 = 11.212</li>
                    <li><strong>Step 3:</strong> Compute probabilities</li>
                    <li className="ml-4">P(y_0) = exp(2.0) / 11.212 = 7.389 / 11.212 ≈ 0.659 (65.9%)</li>
                    <li className="ml-4">P(y_1) = exp(1.0) / 11.212 = 2.718 / 11.212 ≈ 0.243 (24.3%)</li>
                    <li className="ml-4">P(y_2) = exp(0.1) / 11.212 = 1.105 / 11.212 ≈ 0.099 (9.9%)</li>
                    <li>Check: 0.659 + 0.243 + 0.099 ≈ 1.0 ✓ (probabilities sum to 1)</li>
                    <li>Higher logit → Higher probability</li>
                    <li>All probabilities are positive and sum to 1</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Integrated:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Word embeddings, matrix operations, attention matrices, cosine similarity</li>
              <li><strong>Calculus:</strong> Gradient descent, backpropagation, chain rule, optimization</li>
              <li><strong>Probability:</strong> Softmax distributions, sampling strategies, temperature scaling</li>
              <li><strong>Neural Networks:</strong> LSTM cells, Transformer architecture, encoder-decoder structure</li>
              <li><strong>NLP:</strong> Tokenization, sequence-to-sequence models, language modeling</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Learning Outcomes:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Understand how mathematical concepts translate to code</li>
              <li>See formulas applied in real neural network architectures</li>
              <li>Learn complete ML pipeline from data to deployment</li>
              <li>Gain hands-on experience building a production-ready system</li>
              <li>Connect theory with practical implementation</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'maze-solver' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Maze Solver - Pathfinding Algorithms</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Learn Pathfinding Algorithms?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Pathfinding algorithms are fundamental to AI and robotics. Use them when you need to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Find optimal paths in games and navigation systems</li>
                <li>Solve routing problems (GPS, logistics, network routing)</li>
                <li>Understand graph theory and search algorithms</li>
                <li>Build AI agents that navigate environments</li>
                <li>Optimize resource allocation and planning</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How Pathfinding Algorithms Work:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Represent Problem:</strong> Convert maze to graph (nodes = cells, edges = connections)</li>
                <li><strong>Choose Algorithm:</strong> Select A*, BFS, DFS, or Dijkstra based on requirements</li>
                <li><strong>Initialize:</strong> Set start node, end node, and data structures (queue, stack, priority queue)</li>
                <li><strong>Search:</strong> Explore nodes systematically, tracking visited nodes and paths</li>
                <li><strong>Heuristic (A*):</strong> Use Manhattan distance to guide search toward goal</li>
                <li><strong>Path Reconstruction:</strong> Backtrack from end to start using parent pointers</li>
                <li><strong>Optimization:</strong> A* finds optimal path efficiently using f(n) = g(n) + h(n)</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* A* Evaluation Function */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. A* Evaluation Function:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">f(n) = g(n) + h(n)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>f(n)</strong>: Total estimated cost from start to goal through node n</li>
                    <li><strong>g(n)</strong>: Actual cost from start to node n (known, measured)</li>
                    <li><strong>h(n)</strong>: Heuristic estimate from node n to goal (estimated)</li>
                    <li><strong>n</strong>: Current node being evaluated</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Start position: (0, 0)</li>
                    <li>Current node n: (2, 3)</li>
                    <li>Goal position: (5, 7)</li>
                    <li>Each step costs: 1</li>
                    <li><strong>Step 1:</strong> Calculate g(n) - actual cost from start</li>
                    <li className="ml-4">Path: (0,0) → (1,0) → (2,0) → (2,1) → (2,2) → (2,3)</li>
                    <li className="ml-4">Number of steps: 5</li>
                    <li className="ml-4">g(n) = 5</li>
                    <li><strong>Step 2:</strong> Calculate h(n) - heuristic estimate to goal</li>
                    <li className="ml-4">Using Manhattan: h(n) = |2-5| + |3-7| = 3 + 4 = 7</li>
                    <li><strong>Step 3:</strong> Calculate f(n)</li>
                    <li className="ml-4">f(n) = g(n) + h(n) = 5 + 7 = 12</li>
                    <li>Interpretation: We've traveled 5 steps, estimate 7 more steps needed</li>
                  </ul>
                </div>
              </div>

              {/* Manhattan Distance */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Manhattan Distance Heuristic:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">h(n) = |x<sub>1</sub> - x<sub>2</sub>| + |y<sub>1</sub> - y<sub>2</sub>|</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>h(n)</strong>: Manhattan distance (heuristic value)</li>
                    <li><strong>x<sub>1</sub>, y<sub>1</sub></strong>: Coordinates of current node n</li>
                    <li><strong>x<sub>2</sub>, y<sub>2</sub></strong>: Coordinates of goal node</li>
                    <li><strong>| |</strong>: Absolute value (always positive)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current node: (4, 2)</li>
                    <li>Goal node: (8, 6)</li>
                    <li><strong>Step 1:</strong> Calculate x difference</li>
                    <li className="ml-4">|x<sub>1</sub> - x<sub>2</sub>| = |4 - 8| = |−4| = 4</li>
                    <li><strong>Step 2:</strong> Calculate y difference</li>
                    <li className="ml-4">|y<sub>1</sub> - y<sub>2</sub>| = |2 - 6| = |−4| = 4</li>
                    <li><strong>Step 3:</strong> Sum the differences</li>
                    <li className="ml-4">h(n) = 4 + 4 = 8</li>
                    <li>Meaning: At least 8 steps needed (4 right + 4 down)</li>
                    <li>Perfect for 4-directional movement (up/down/left/right only)</li>
                  </ul>
                </div>
              </div>

              {/* Euclidean Distance */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">3. Euclidean Distance:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">h(n) = √((x<sub>1</sub> - x<sub>2</sub>)² + (y<sub>1</sub> - y<sub>2</sub>)²)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>h(n)</strong>: Euclidean distance (straight-line distance)</li>
                    <li><strong>x<sub>1</sub>, y<sub>1</sub></strong>: Coordinates of current node</li>
                    <li><strong>x<sub>2</sub>, y<sub>2</sub></strong>: Coordinates of goal node</li>
                    <li><strong>√</strong>: Square root</li>
                    <li><strong>²</strong>: Squared (multiply by itself)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current node: (3, 4)</li>
                    <li>Goal node: (7, 1)</li>
                    <li><strong>Step 1:</strong> Calculate x difference and square it</li>
                    <li className="ml-4">(x<sub>1</sub> - x<sub>2</sub>)² = (3 - 7)² = (−4)² = 16</li>
                    <li><strong>Step 2:</strong> Calculate y difference and square it</li>
                    <li className="ml-4">(y<sub>1</sub> - y<sub>2</sub>)² = (4 - 1)² = (3)² = 9</li>
                    <li><strong>Step 3:</strong> Sum the squares</li>
                    <li className="ml-4">16 + 9 = 25</li>
                    <li><strong>Step 4:</strong> Take square root</li>
                    <li className="ml-4">h(n) = √25 = 5</li>
                    <li>Meaning: Straight-line distance is 5 units</li>
                    <li>Used for 8-directional movement (can move diagonally)</li>
                  </ul>
                </div>
              </div>

              {/* Dijkstra Distance Update */}
              <div>
                <strong className="text-gray-800 text-base">4. Dijkstra Distance Update:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">dist[v] = min(dist[v], dist[u] + weight(u, v))</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>dist[v]</strong>: Current shortest distance to node v</li>
                    <li><strong>dist[u]</strong>: Shortest distance to node u (already known)</li>
                    <li><strong>weight(u, v)</strong>: Cost to travel from node u to node v</li>
                    <li><strong>min(...)</strong>: Keep the smaller value (shorter path)</li>
                    <li><strong>u</strong>: Current node being processed</li>
                    <li><strong>v</strong>: Neighbor node being updated</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current node u: A (dist[A] = 3)</li>
                    <li>Neighbor node v: B</li>
                    <li>Edge weight: weight(A, B) = 5</li>
                    <li>Current dist[B] = 10 (from previous path)</li>
                    <li><strong>Step 1:</strong> Calculate new distance via u</li>
                    <li className="ml-4">dist[u] + weight(u, v) = 3 + 5 = 8</li>
                    <li><strong>Step 2:</strong> Compare with current distance</li>
                    <li className="ml-4">min(dist[v], new_distance) = min(10, 8) = 8</li>
                    <li><strong>Step 3:</strong> Update distance</li>
                    <li className="ml-4">dist[B] = 8 (found shorter path!)</li>
                    <li>If dist[B] was 7, it would stay 7 (7 &lt; 8)</li>
                    <li>Always keeps the minimum distance found so far</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Algorithm Comparison:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>A*:</strong> Optimal path, uses heuristic, best for most cases (O(b^d) time)</li>
              <li><strong>BFS:</strong> Guarantees shortest path in unweighted graphs, explores level by level (O(V + E))</li>
              <li><strong>DFS:</strong> Fast but may not find shortest path, uses less memory (O(V + E))</li>
              <li><strong>Dijkstra:</strong> Optimal for weighted graphs, no heuristic needed (O((V + E) log V))</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Graph Theory:</strong> Represent maze as graph with nodes and edges</li>
              <li><strong>Heuristics:</strong> Estimate distance to goal (admissible heuristics ensure optimality)</li>
              <li><strong>Priority Queues:</strong> Efficiently select next node to explore</li>
              <li><strong>Dynamic Programming:</strong> Store and reuse computed distances</li>
              <li><strong>Backtracking:</strong> Reconstruct path from goal to start</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Game AI (NPC pathfinding, strategy games)</li>
              <li>GPS navigation systems (route planning)</li>
              <li>Robotics (robot navigation, obstacle avoidance)</li>
              <li>Network routing (packet routing, internet protocols)</li>
              <li>Logistics and delivery optimization</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'gradient-descent' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Gradient Descent Visualization</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Understand Gradient Descent?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Gradient descent is the foundation of optimization in machine learning. Understanding it helps you:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Understand how neural networks learn and optimize</li>
                <li>Choose appropriate learning rates for training</li>
                <li>Debug training issues (convergence, overshooting, divergence)</li>
                <li>Grasp the connection between Calculus and ML optimization</li>
                <li>Visualize the optimization process in action</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How Gradient Descent Works:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Start at Initial Point:</strong> Begin with random or initial parameter values</li>
                <li><strong>Calculate Gradient:</strong> Compute derivative f'(x) at current point</li>
                <li><strong>Update Position:</strong> Move in direction opposite to gradient: x = x - α·f'(x)</li>
                <li><strong>Repeat:</strong> Continue until gradient is near zero (convergence)</li>
                <li><strong>Learning Rate:</strong> α controls step size - too small (slow), too large (overshoot)</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Single Variable */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Gradient Descent Update (Single Variable):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">x<sub>{'{'}n+1{'}'}</sub> = x<sub>n</sub> - α · f'(x<sub>n</sub>)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>x<sub>{'{'}n+1{'}'}</sub></strong>: Next position</li>
                    <li><strong>x<sub>n</sub></strong>: Current position</li>
                    <li><strong>α</strong>: Learning rate (step size)</li>
                    <li><strong>f'(x)</strong>: Derivative (gradient) at current point</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Function: f(x) = x² - 4x + 3 (minimum at x = 2)</li>
                    <li>Derivative: f'(x) = 2x - 4</li>
                    <li>Start: x₀ = 5, Learning rate: α = 0.1</li>
                    <li><strong>Iteration 1:</strong></li>
                    <li className="ml-4">f'(5) = 2×5 - 4 = 6</li>
                    <li className="ml-4">x₁ = 5 - 0.1 × 6 = 5 - 0.6 = 4.4</li>
                    <li><strong>Iteration 2:</strong></li>
                    <li className="ml-4">f'(4.4) = 2×4.4 - 4 = 4.8</li>
                    <li className="ml-4">x₂ = 4.4 - 0.1 × 4.8 = 4.4 - 0.48 = 3.92</li>
                    <li>Continues until x ≈ 2.0 (minimum)</li>
                  </ul>
                </div>
              </div>

              {/* Multiple Parameters */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Gradient Descent (Multiple Parameters):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ<sub>{'{'}n+1{'}'}</sub> = θ<sub>n</sub> - α · ∇<sub>θ</sub>L(θ<sub>n</sub>)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>θ</strong>: Parameter vector [θ₁, θ₂, ..., θₙ]</li>
                    <li><strong>∇<sub>θ</sub>L</strong>: Gradient vector [∂L/∂θ₁, ∂L/∂θ₂, ...]</li>
                    <li><strong>α</strong>: Learning rate</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Loss function: L(θ₁, θ₂) = θ₁² + θ₂² (minimum at (0,0))</li>
                    <li>Gradients: ∇L = [2θ₁, 2θ₂]</li>
                    <li>Start: θ₀ = [1.5, 2.0], Learning rate: α = 0.1</li>
                    <li><strong>Iteration 1:</strong></li>
                    <li className="ml-4">∇L(1.5, 2.0) = [3.0, 4.0]</li>
                    <li className="ml-4">θ₁ = [1.5, 2.0] - 0.1 × [3.0, 4.0] = [1.5, 2.0] - [0.3, 0.4] = [1.2, 1.6]</li>
                    <li><strong>Iteration 2:</strong></li>
                    <li className="ml-4">∇L(1.2, 1.6) = [2.4, 3.2]</li>
                    <li className="ml-4">θ₂ = [1.2, 1.6] - 0.1 × [2.4, 3.2] = [0.96, 1.28]</li>
                    <li>Converges to [0, 0]</li>
                  </ul>
                </div>
              </div>

              {/* Convergence */}
              <div>
                <strong className="text-gray-800 text-base">3. Convergence Condition:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">|f'(x)| &lt; ε</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>|f'(x)|</strong>: Absolute value of gradient</li>
                    <li><strong>ε</strong>: Small threshold (e.g., 0.001, 0.0001)</li>
                    <li>When gradient is near zero, we're at a minimum</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Threshold: ε = 0.001</li>
                    <li>At x = 2.001: f'(2.001) = 2×2.001 - 4 = 0.002</li>
                    <li>|f'(2.001)| = 0.002 &gt; 0.001 (not converged yet)</li>
                    <li>At x = 2.0005: f'(2.0005) = 2×2.0005 - 4 = 0.001</li>
                    <li>|f'(2.0005)| = 0.001 = ε (converged!)</li>
                    <li>Algorithm stops when gradient is small enough</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Calculus:</strong> Derivatives measure rate of change, gradients point to steepest ascent</li>
              <li><strong>Optimization:</strong> Finding minimum/maximum of functions</li>
              <li><strong>Learning Rate:</strong> Hyperparameter controlling step size</li>
              <li><strong>Convergence:</strong> When algorithm reaches optimal solution</li>
              <li><strong>Local vs Global Minima:</strong> Gradient descent may find local minimum</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Training neural networks (backpropagation uses gradient descent)</li>
              <li>Linear regression optimization</li>
              <li>Logistic regression parameter estimation</li>
              <li>Support vector machines</li>
              <li>Any ML model that minimizes a loss function</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'neural-network-playground' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Neural Network Playground</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Visualize Neural Networks?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Seeing how neural networks work helps you understand the core concepts:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>Understand forward pass: how data flows through layers</li>
                <li>See backpropagation: how errors propagate backward</li>
                <li>Visualize weight updates: how network learns</li>
                <li>Connect Linear Algebra and Calculus to neural networks</li>
                <li>Debug network behavior and understand activations</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How Neural Networks Learn:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Forward Pass:</strong> Input flows through layers, each applies W·a + b then activation</li>
                <li><strong>Calculate Loss:</strong> Compare prediction to target using loss function</li>
                <li><strong>Backward Pass:</strong> Calculate gradients using chain rule, propagate errors backward</li>
                <li><strong>Update Weights:</strong> Adjust weights using gradient descent: w = w - α·∂L/∂w</li>
                <li><strong>Repeat:</strong> Continue training until loss is minimized</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Forward Pass */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Forward Pass (Linear Algebra):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">z = W · a + b</p>
                <p className="text-gray-700 text-sm font-mono">a = σ(z)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>z</strong>: Pre-activation (weighted sum before activation)</li>
                    <li><strong>W</strong>: Weight matrix (learned parameters)</li>
                    <li><strong>a</strong>: Activation vector (input from previous layer)</li>
                    <li><strong>b</strong>: Bias vector (learned offset)</li>
                    <li><strong>σ</strong>: Activation function (e.g., sigmoid, ReLU, tanh)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Input layer: 3 neurons, Output layer: 2 neurons</li>
                    <li>Weight matrix W: 2 × 3 = [[0.5, -0.3, 0.8], [0.2, 0.6, -0.4]]</li>
                    <li>Input activations a = [0.7, 0.3, 0.9]</li>
                    <li>Bias b = [0.1, -0.2]</li>
                    <li><strong>Step 1:</strong> Compute z = W · a + b</li>
                    <li className="ml-4">z[0] = (0.5×0.7 + -0.3×0.3 + 0.8×0.9) + 0.1 = (0.35 - 0.09 + 0.72) + 0.1 = 1.08</li>
                    <li className="ml-4">z[1] = (0.2×0.7 + 0.6×0.3 + -0.4×0.9) + (-0.2) = (0.14 + 0.18 - 0.36) - 0.2 = -0.24</li>
                    <li><strong>Step 2:</strong> Apply activation (sigmoid: σ(z) = 1/(1+e^(-z)))</li>
                    <li className="ml-4">a[0] = σ(1.08) = 1/(1+e^(-1.08)) ≈ 0.746</li>
                    <li className="ml-4">a[1] = σ(-0.24) = 1/(1+e^(0.24)) ≈ 0.440</li>
                    <li>Output: a = [0.746, 0.440]</li>
                  </ul>
                </div>
              </div>

              {/* Backward Pass */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Backward Pass (Calculus - Chain Rule):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>∂L/∂w</strong>: Gradient of loss with respect to weight (what we want)</li>
                    <li><strong>∂L/∂a</strong>: Gradient of loss with respect to activation (from next layer)</li>
                    <li><strong>∂a/∂z</strong>: Derivative of activation function</li>
                    <li><strong>∂z/∂w</strong>: Derivative of weighted sum with respect to weight</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Loss L = 0.5, Activation a = 0.746, Weight w = 0.5</li>
                    <li>Using sigmoid: σ'(z) = σ(z)(1 - σ(z))</li>
                    <li><strong>Step 1:</strong> ∂L/∂a = -0.254 (from loss function derivative)</li>
                    <li><strong>Step 2:</strong> ∂a/∂z = σ'(1.08) = 0.746 × (1 - 0.746) = 0.746 × 0.254 ≈ 0.189</li>
                    <li><strong>Step 3:</strong> ∂z/∂w = a_input = 0.7 (input that was multiplied by w)</li>
                    <li><strong>Step 4:</strong> Combine using chain rule:</li>
                    <li className="ml-4">∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w</li>
                    <li className="ml-4">∂L/∂w = (-0.254) × 0.189 × 0.7 ≈ -0.034</li>
                    <li>Gradient tells us: increasing w will decrease loss (negative gradient)</li>
                  </ul>
                </div>
              </div>

              {/* Weight Update */}
              <div>
                <strong className="text-gray-800 text-base">3. Weight Update (Gradient Descent):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">w<sub>{'{'}new{'}'}</sub> = w<sub>old</sub> - α · ∂L/∂w</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>w<sub>{'{'}new{'}'}</sub></strong>: Updated weight after gradient descent step</li>
                    <li><strong>w<sub>old</sub></strong>: Current weight value</li>
                    <li><strong>α</strong>: Learning rate (step size, e.g., 0.01, 0.001)</li>
                    <li><strong>∂L/∂w</strong>: Gradient computed from backpropagation</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current weight: w_old = 0.5</li>
                    <li>Learning rate: α = 0.01</li>
                    <li>Gradient: ∂L/∂w = -0.034</li>
                    <li><strong>Update:</strong></li>
                    <li className="ml-4">w_new = w_old - α · ∂L/∂w</li>
                    <li className="ml-4">w_new = 0.5 - 0.01 × (-0.034)</li>
                    <li className="ml-4">w_new = 0.5 + 0.00034 = 0.50034</li>
                    <li>Weight increased slightly (moving in direction opposite to gradient)</li>
                    <li>After many iterations, weights converge to optimal values</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Integrated:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Matrix multiplication (W·a), vector addition (bias)</li>
              <li><strong>Calculus:</strong> Chain rule for backpropagation, derivatives for gradients</li>
              <li><strong>Probability:</strong> Activation functions map to probability-like outputs</li>
              <li><strong>Optimization:</strong> Gradient descent minimizes loss function</li>
              <li><strong>Neural Networks:</strong> Layers, neurons, weights, biases, activations</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Image classification (CNNs use same principles)</li>
              <li>Natural language processing (RNNs, Transformers)</li>
              <li>Regression and prediction tasks</li>
              <li>Pattern recognition</li>
              <li>Any supervised learning problem</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'linear-regression' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Linear Regression Visualization</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Understand Linear Regression?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Linear regression is the foundation of many ML algorithms. It demonstrates:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>How Linear Algebra (matrix operations) and Calculus (gradients) work together</li>
                <li>The fundamental optimization process used in all ML</li>
                <li>Least squares method and error minimization</li>
                <li>Gradient descent in action on real data</li>
                <li>Basis for understanding more complex models</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How Linear Regression Works:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Model:</strong> y = mx + b (line equation)</li>
                <li><strong>Loss Function:</strong> MSE = (1/m) Σ(y_pred - y_true)²</li>
                <li><strong>Gradients:</strong> Calculate ∂L/∂m and ∂L/∂b using calculus</li>
                <li><strong>Update:</strong> m = m - α·∂L/∂m, b = b - α·∂L/∂b</li>
                <li><strong>Repeat:</strong> Until loss is minimized</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Prediction */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Prediction:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">ŷ = m·x + b</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>ŷ</strong>: Predicted output value</li>
                    <li><strong>m</strong>: Slope (weight/coefficient)</li>
                    <li><strong>x</strong>: Input feature value</li>
                    <li><strong>b</strong>: Intercept (bias term)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Slope m = 2.5, Intercept b = 1.0</li>
                    <li>Input x = 3.0</li>
                    <li><strong>Prediction:</strong></li>
                    <li className="ml-4">ŷ = m·x + b = 2.5 × 3.0 + 1.0 = 7.5 + 1.0 = 8.5</li>
                    <li>For x = 5.0: ŷ = 2.5 × 5.0 + 1.0 = 13.5</li>
                    <li>For x = 0.0: ŷ = 2.5 × 0.0 + 1.0 = 1.0 (intercept point)</li>
                  </ul>
                </div>
              </div>

              {/* Loss Function */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Loss Function (MSE - Mean Squared Error):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">L = (1/m) Σ(ŷ<sub>i</sub> - y<sub>i</sub>)²</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>L</strong>: Total loss (average squared error)</li>
                    <li><strong>m</strong>: Number of data points</li>
                    <li><strong>ŷ<sub>i</sub></strong>: Predicted value for data point i</li>
                    <li><strong>y<sub>i</sub></strong>: True/actual value for data point i</li>
                    <li><strong>Σ</strong>: Sum over all data points</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>3 data points: (x₁=1, y₁=3), (x₂=2, y₂=6), (x₃=3, y₃=8)</li>
                    <li>Model: ŷ = 2.5x + 1.0</li>
                    <li><strong>Predictions:</strong></li>
                    <li className="ml-4">ŷ₁ = 2.5×1 + 1.0 = 3.5</li>
                    <li className="ml-4">ŷ₂ = 2.5×2 + 1.0 = 6.0</li>
                    <li className="ml-4">ŷ₃ = 2.5×3 + 1.0 = 8.5</li>
                    <li><strong>Errors:</strong></li>
                    <li className="ml-4">(ŷ₁ - y₁)² = (3.5 - 3)² = 0.5² = 0.25</li>
                    <li className="ml-4">(ŷ₂ - y₂)² = (6.0 - 6)² = 0² = 0.0</li>
                    <li className="ml-4">(ŷ₃ - y₃)² = (8.5 - 8)² = 0.5² = 0.25</li>
                    <li><strong>Loss:</strong></li>
                    <li className="ml-4">L = (1/3) × (0.25 + 0.0 + 0.25) = (1/3) × 0.5 ≈ 0.167</li>
                    <li>Lower loss = better fit</li>
                  </ul>
                </div>
              </div>

              {/* Gradients */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">3. Gradients:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">∂L/∂m = (2/m) Σ(ŷ<sub>i</sub> - y<sub>i</sub>) · x<sub>i</sub></p>
                <p className="text-gray-700 text-sm font-mono">∂L/∂b = (2/m) Σ(ŷ<sub>i</sub> - y<sub>i</sub>)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>∂L/∂m</strong>: Gradient with respect to slope (how loss changes with m)</li>
                    <li><strong>∂L/∂b</strong>: Gradient with respect to intercept (how loss changes with b)</li>
                    <li><strong>2/m</strong>: Constant from derivative of squared error</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Same data: (1,3), (2,6), (3,8) with predictions 3.5, 6.0, 8.5</li>
                    <li><strong>Gradient for slope m:</strong></li>
                    <li className="ml-4">∂L/∂m = (2/3) × [(3.5-3)×1 + (6.0-6)×2 + (8.5-8)×3]</li>
                    <li className="ml-4">∂L/∂m = (2/3) × [0.5×1 + 0×2 + 0.5×3]</li>
                    <li className="ml-4">∂L/∂m = (2/3) × [0.5 + 0 + 1.5] = (2/3) × 2.0 ≈ 1.333</li>
                    <li><strong>Gradient for intercept b:</strong></li>
                    <li className="ml-4">∂L/∂b = (2/3) × [(3.5-3) + (6.0-6) + (8.5-8)]</li>
                    <li className="ml-4">∂L/∂b = (2/3) × [0.5 + 0 + 0.5] = (2/3) × 1.0 ≈ 0.667</li>
                    <li>Positive gradients mean increasing m/b increases loss (need to decrease)</li>
                  </ul>
                </div>
              </div>

              {/* Gradient Descent */}
              <div>
                <strong className="text-gray-800 text-base">4. Gradient Descent Update:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">θ = θ - α · ∇L</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>θ</strong>: Parameter vector [b, m]</li>
                    <li><strong>α</strong>: Learning rate (step size, e.g., 0.01)</li>
                    <li><strong>∇L</strong>: Gradient vector [∂L/∂b, ∂L/∂m]</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current: m = 2.5, b = 1.0</li>
                    <li>Learning rate: α = 0.01</li>
                    <li>Gradients: ∂L/∂m = 1.333, ∂L/∂b = 0.667</li>
                    <li><strong>Update:</strong></li>
                    <li className="ml-4">m_new = m - α · ∂L/∂m = 2.5 - 0.01 × 1.333 = 2.5 - 0.01333 ≈ 2.487</li>
                    <li className="ml-4">b_new = b - α · ∂L/∂b = 1.0 - 0.01 × 0.667 = 1.0 - 0.00667 ≈ 0.993</li>
                    <li>After many iterations, m and b converge to optimal values</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Matrix multiplication for predictions, vector operations</li>
              <li><strong>Calculus:</strong> Partial derivatives, gradients, chain rule</li>
              <li><strong>Statistics:</strong> Mean squared error, least squares method</li>
              <li><strong>Optimization:</strong> Gradient descent to minimize loss</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Predicting house prices from features</li>
              <li>Sales forecasting</li>
              <li>Stock price prediction</li>
              <li>Any continuous value prediction</li>
              <li>Foundation for neural networks</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'pca-visualization' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">PCA Visualization</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Understand PCA?</h4>
              <p className="text-blue-800 text-sm mb-2">
                PCA directly applies Linear Algebra concepts you learned:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>See eigenvalues and eigenvectors in action</li>
                <li>Understand covariance matrices and their meaning</li>
                <li>Visualize dimensionality reduction</li>
                <li>Connect theory to practical data analysis</li>
                <li>Foundation for many ML preprocessing techniques</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How PCA Works:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Center Data:</strong> Subtract mean from each feature</li>
                <li><strong>Covariance Matrix:</strong> Calculate C = (1/n) X^T · X</li>
                <li><strong>Eigenvalue Decomposition:</strong> Find eigenvalues λ and eigenvectors v where C·v = λ·v</li>
                <li><strong>Select Components:</strong> Choose eigenvectors with largest eigenvalues</li>
                <li><strong>Project:</strong> Transform data: Y = X · W (W = eigenvectors)</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Covariance Matrix */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Covariance Matrix:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">C = (1/n) X^T · X</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>C</strong>: Covariance matrix (d×d where d = number of features)</li>
                    <li><strong>X</strong>: Centered data matrix (n×d, n = samples, d = features)</li>
                    <li><strong>X^T</strong>: Transpose of X (d×n)</li>
                    <li><strong>n</strong>: Number of data points</li>
                    <li><strong>1/n</strong>: Normalization factor</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>3 data points, 2 features: X = [[1, 2], [3, 4], [5, 6]]</li>
                    <li><strong>Step 1:</strong> Center data (subtract mean)</li>
                    <li className="ml-4">Mean = [3, 4]</li>
                    <li className="ml-4">X_centered = [[-2, -2], [0, 0], [2, 2]]</li>
                    <li><strong>Step 2:</strong> Compute X^T · X</li>
                    <li className="ml-4">X^T = [[-2, 0, 2], [-2, 0, 2]]</li>
                    <li className="ml-4">X^T · X = [[8, 8], [8, 8]]</li>
                    <li><strong>Step 3:</strong> Normalize</li>
                    <li className="ml-4">C = (1/3) × [[8, 8], [8, 8]] = [[2.67, 2.67], [2.67, 2.67]]</li>
                    <li>Covariance shows features vary together (positive correlation)</li>
                  </ul>
                </div>
              </div>

              {/* Eigenvalue Equation */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Eigenvalue Equation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">C · v = λ · v</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>C</strong>: Covariance matrix</li>
                    <li><strong>v</strong>: Eigenvector (direction of maximum variance)</li>
                    <li><strong>λ</strong>: Eigenvalue (amount of variance in direction v)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>C = [[2.67, 2.67], [2.67, 2.67]]</li>
                    <li><strong>Eigenvalue λ₁ = 5.33:</strong></li>
                    <li className="ml-4">Eigenvector v₁ = [0.707, 0.707] (normalized)</li>
                    <li className="ml-4">Check: C · v₁ = [[2.67, 2.67], [2.67, 2.67]] · [0.707, 0.707]</li>
                    <li className="ml-4">C · v₁ = [3.77, 3.77] = 5.33 × [0.707, 0.707] = λ₁ · v₁ ✓</li>
                    <li><strong>Eigenvalue λ₂ = 0:</strong></li>
                    <li className="ml-4">Eigenvector v₂ = [-0.707, 0.707]</li>
                    <li>Larger eigenvalue = more variance in that direction</li>
                  </ul>
                </div>
              </div>

              {/* Projection */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">3. Projection:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Y = X · W</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Y</strong>: Projected data (n×k, k = number of components)</li>
                    <li><strong>X</strong>: Original centered data (n×d)</li>
                    <li><strong>W</strong>: Matrix of eigenvectors (d×k)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>X_centered = [[-2, -2], [0, 0], [2, 2]]</li>
                    <li>W = [v₁] = [[0.707], [0.707]] (using only first PC)</li>
                    <li><strong>Projection:</strong></li>
                    <li className="ml-4">Y = X · W = [[-2, -2], [0, 0], [2, 2]] · [[0.707], [0.707]]</li>
                    <li className="ml-4">Y = [[-2.828], [0], [2.828]]</li>
                    <li>Reduced from 2D to 1D while preserving maximum variance</li>
                  </ul>
                </div>
              </div>

              {/* Variance Explained */}
              <div>
                <strong className="text-gray-800 text-base">4. Variance Explained:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Variance<sub>i</sub> = λ<sub>i</sub> / Σλ</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Variance<sub>i</sub></strong>: Proportion of variance explained by component i</li>
                    <li><strong>λ<sub>i</sub></strong>: Eigenvalue of component i</li>
                    <li><strong>Σλ</strong>: Sum of all eigenvalues</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Eigenvalues: λ₁ = 5.33, λ₂ = 0</li>
                    <li>Total variance: Σλ = 5.33 + 0 = 5.33</li>
                    <li><strong>Variance explained:</strong></li>
                    <li className="ml-4">PC1: 5.33 / 5.33 = 1.0 = 100%</li>
                    <li className="ml-4">PC2: 0 / 5.33 = 0%</li>
                    <li>First component captures all variance (perfect correlation)</li>
                    <li>In practice, we keep components explaining most variance (e.g., 95%)</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Eigenvalues, eigenvectors, matrix multiplication, covariance</li>
              <li><strong>Statistics:</strong> Variance, covariance, mean centering</li>
              <li><strong>Dimensionality Reduction:</strong> Reducing features while preserving variance</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Image compression</li>
              <li>Feature extraction before classification</li>
              <li>Data visualization (reduce to 2D/3D)</li>
              <li>Noise reduction</li>
              <li>Preprocessing for ML models</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'convolution-visualization' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Convolution Operation Visualization</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Understand Convolution?</h4>
              <p className="text-blue-800 text-sm mb-2">
                Convolution is the core operation in CNNs. Understanding it helps you:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>See how matrix operations work in image processing</li>
                <li>Understand how CNNs extract features from images</li>
                <li>Grasp the mathematical foundation of computer vision</li>
                <li>Visualize sliding window operations</li>
                <li>Connect Linear Algebra to deep learning architectures</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How Convolution Works:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Kernel:</strong> Small matrix (e.g., 3×3) with learned weights</li>
                <li><strong>Slide:</strong> Move kernel over input matrix</li>
                <li><strong>Multiply:</strong> Element-wise multiplication of overlapping regions</li>
                <li><strong>Sum:</strong> Add all products to get output value</li>
                <li><strong>Repeat:</strong> For each position in output</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Convolution Operation */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Convolution Operation:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">(I * K)[i,j] = Σ<sub>m</sub> Σ<sub>n</sub> I[i+m, j+n] · K[m, n]</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>(I * K)[i,j]</strong>: Output value at position (i,j)</li>
                    <li><strong>I</strong>: Input matrix/image</li>
                    <li><strong>K</strong>: Kernel/filter matrix</li>
                    <li><strong>m, n</strong>: Indices over kernel dimensions</li>
                    <li><strong>Σ</strong>: Sum over all kernel positions</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Input I (4×4): [[1, 1, 1, 1], [1, 5, 5, 1], [1, 5, 5, 1], [1, 1, 1, 1]]</li>
                    <li>Kernel K (3×3): [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] (edge detection)</li>
                    <li><strong>Compute output[0,0]:</strong></li>
                    <li className="ml-4">Overlap: I[0:3, 0:3] = [[1, 1, 1], [1, 5, 5], [1, 5, 5]]</li>
                    <li className="ml-4">(I * K)[0,0] = 1×(-1) + 1×(-1) + 1×(-1) + 1×(-1) + 5×8 + 5×(-1) + 1×(-1) + 5×(-1) + 5×(-1)</li>
                    <li className="ml-4">(I * K)[0,0] = -1 -1 -1 -1 + 40 -5 -1 -5 -5 = 20</li>
                    <li><strong>Compute output[1,1]:</strong></li>
                    <li className="ml-4">Overlap: I[1:4, 1:4] = [[5, 5, 1], [5, 5, 1], [1, 1, 1]]</li>
                    <li className="ml-4">(I * K)[1,1] = 5×(-1) + 5×(-1) + 1×(-1) + 5×(-1) + 5×8 + 1×(-1) + 1×(-1) + 1×(-1) + 1×(-1)</li>
                    <li className="ml-4">(I * K)[1,1] = -5 -5 -1 -5 + 40 -1 -1 -1 -1 = 20</li>
                    <li>High values indicate edges (transitions from 1 to 5)</li>
                  </ul>
                </div>
              </div>

              {/* Output Size */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Output Size (No Padding):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Output = Input - Kernel + 1</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Input</strong>: Size of input matrix (height or width)</li>
                    <li><strong>Kernel</strong>: Size of kernel (height or width)</li>
                    <li><strong>Output</strong>: Size of output matrix</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Input: 4×4, Kernel: 3×3</li>
                    <li>Output height = 4 - 3 + 1 = 2</li>
                    <li>Output width = 4 - 3 + 1 = 2</li>
                    <li>Output: 2×2 matrix</li>
                    <li>Each dimension shrinks by (kernel_size - 1)</li>
                  </ul>
                </div>
              </div>

              {/* With Padding */}
              <div>
                <strong className="text-gray-800 text-base">3. Output Size (With Padding):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Output = Input - Kernel + 2·Padding + 1</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Padding</strong>: Number of zeros added around input (e.g., 1, 2)</li>
                    <li><strong>2·Padding</strong>: Padding added on both sides</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Input: 4×4, Kernel: 3×3, Padding: 1</li>
                    <li>Padded input: 6×6 (4 + 2×1)</li>
                    <li>Output = 4 - 3 + 2×1 + 1 = 4</li>
                    <li>Output: 4×4 (same size as input!)</li>
                    <li>Padding preserves spatial dimensions</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Matrix multiplication, element-wise operations, sliding windows</li>
              <li><strong>Feature Extraction:</strong> Kernels detect edges, textures, patterns</li>
              <li><strong>CNNs:</strong> Foundation of convolutional neural networks</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Image classification (ResNet, VGG)</li>
              <li>Object detection (YOLO, R-CNN)</li>
              <li>Image segmentation</li>
              <li>Medical image analysis</li>
              <li>Any computer vision task</li>
            </ul>
          </div>
        </div>
      )}

      {selectedApplication === 'k-means-clustering' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">K-Means Clustering Visualization</h3>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">Why Understand K-Means?</h4>
              <p className="text-blue-800 text-sm mb-2">
                K-Means is a fundamental unsupervised learning algorithm that demonstrates:
              </p>
              <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
                <li>How distance metrics (Linear Algebra) group similar data</li>
                <li>How statistics (mean/centroid) represent clusters</li>
                <li>Iterative optimization process</li>
                <li>Foundation for many clustering algorithms</li>
                <li>Practical applications in data analysis</li>
              </ul>
            </div>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">How K-Means Works:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Initialize:</strong> Choose k centroids (K-means++ for better initialization)</li>
                <li><strong>Assign:</strong> Each point assigned to nearest centroid using Euclidean distance</li>
                <li><strong>Update:</strong> Centroids moved to mean of assigned points</li>
                <li><strong>Repeat:</strong> Steps 2-3 until centroids converge (don't move much)</li>
                <li><strong>Result:</strong> k clusters with minimized within-cluster variance</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Euclidean Distance */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">1. Euclidean Distance (Linear Algebra):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">d(x, c) = √(Σ(xᵢ - cᵢ)²)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>d(x, c)</strong>: Distance between point x and centroid c</li>
                    <li><strong>xᵢ</strong>: i-th coordinate of point x</li>
                    <li><strong>cᵢ</strong>: i-th coordinate of centroid c</li>
                    <li><strong>Σ</strong>: Sum over all dimensions</li>
                    <li><strong>√</strong>: Square root</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Point x = [3, 4], Centroid c = [1, 2]</li>
                    <li><strong>Step 1:</strong> Calculate squared differences</li>
                    <li className="ml-4">(x₁ - c₁)² = (3 - 1)² = 2² = 4</li>
                    <li className="ml-4">(x₂ - c₂)² = (4 - 2)² = 2² = 4</li>
                    <li><strong>Step 2:</strong> Sum squared differences</li>
                    <li className="ml-4">Σ(xᵢ - cᵢ)² = 4 + 4 = 8</li>
                    <li><strong>Step 3:</strong> Take square root</li>
                    <li className="ml-4">d(x, c) = √8 ≈ 2.828</li>
                    <li>Smaller distance = point is closer to centroid</li>
                  </ul>
                </div>
              </div>

              {/* Centroid Update */}
              <div className="border-b pb-3">
                <strong className="text-gray-800 text-base">2. Centroid Update (Statistics):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">c<sub>new</sub> = (1/n) Σ xᵢ</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>c<sub>new</sub></strong>: Updated centroid position</li>
                    <li><strong>n</strong>: Number of points in cluster</li>
                    <li><strong>xᵢ</strong>: i-th point assigned to cluster</li>
                    <li><strong>Σ</strong>: Sum over all points in cluster</li>
                    <li><strong>1/n</strong>: Average (mean) calculation</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Cluster has 3 points: x₁ = [2, 3], x₂ = [4, 5], x₃ = [6, 7]</li>
                    <li><strong>Step 1:</strong> Sum all points</li>
                    <li className="ml-4">Σx = [2+4+6, 3+5+7] = [12, 15]</li>
                    <li><strong>Step 2:</strong> Divide by number of points</li>
                    <li className="ml-4">c<sub>new</sub> = (1/3) × [12, 15] = [4, 5]</li>
                    <li>New centroid is at the mean position of all cluster points</li>
                    <li>This minimizes distance from centroid to all points</li>
                  </ul>
                </div>
              </div>

              {/* WCSS */}
              <div>
                <strong className="text-gray-800 text-base">3. WCSS (Within-Cluster Sum of Squares):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">WCSS = Σ<sub>k</sub> Σ<sub>x∈C<sub>k</sub></sub> ||x - c<sub>k</sub>||²</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>WCSS</strong>: Within-Cluster Sum of Squares (total variance)</li>
                    <li><strong>k</strong>: Cluster index (1, 2, ..., K)</li>
                    <li><strong>C<sub>k</sub></strong>: Set of points in cluster k</li>
                    <li><strong>x</strong>: Point in cluster</li>
                    <li><strong>c<sub>k</sub></strong>: Centroid of cluster k</li>
                    <li><strong>||x - c<sub>k</sub>||²</strong>: Squared Euclidean distance</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>2 clusters, K = 2</li>
                    <li><strong>Cluster 1:</strong> Points [2,3], [4,5], Centroid c₁ = [3,4]</li>
                    <li className="ml-4">WCSS₁ = ||[2,3] - [3,4]||² + ||[4,5] - [3,4]||²</li>
                    <li className="ml-4">WCSS₁ = ||[-1,-1]||² + ||[1,1]||²</li>
                    <li className="ml-4">WCSS₁ = ((-1)² + (-1)²) + (1² + 1²) = 2 + 2 = 4</li>
                    <li><strong>Cluster 2:</strong> Points [8,9], [10,11], Centroid c₂ = [9,10]</li>
                    <li className="ml-4">WCSS₂ = ||[8,9] - [9,10]||² + ||[10,11] - [9,10]||²</li>
                    <li className="ml-4">WCSS₂ = ||[-1,-1]||² + ||[1,1]||² = 2 + 2 = 4</li>
                    <li><strong>Total WCSS:</strong></li>
                    <li className="ml-4">WCSS = WCSS₁ + WCSS₂ = 4 + 4 = 8</li>
                    <li>Lower WCSS = tighter, more compact clusters</li>
                    <li>K-Means minimizes WCSS through iterative updates</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Linear Algebra:</strong> Euclidean distance calculation, vector operations</li>
              <li><strong>Statistics:</strong> Mean calculation for centroids, variance (WCSS)</li>
              <li><strong>Optimization:</strong> Minimize within-cluster variance</li>
              <li><strong>Unsupervised Learning:</strong> No labels needed, finds patterns in data</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Customer segmentation (marketing)</li>
              <li>Image compression (color quantization)</li>
              <li>Document clustering</li>
              <li>Anomaly detection</li>
              <li>Data preprocessing for other ML algorithms</li>
            </ul>
          </div>
        </div>
      )}

      {/* Trading Tools Tutorial */}
      {selectedApplication === 'trading-tools' && (
        <div>
          <h3 className="text-xl font-bold text-gray-900 mb-4">About This Application</h3>
          <div className="space-y-4 text-gray-700">
            <p>
              <strong>AI Trading Tools</strong> is a comprehensive tutorial that teaches you how to build 
              AI-powered trading systems from scratch. This complete step-by-step guide covers everything from 
              data collection to deploying a fully integrated trading system.
            </p>

            <div className="bg-green-50 rounded-lg p-4 mb-4 border-2 border-green-200">
              <h4 className="font-semibold text-green-900 mb-2">Tutorial Structure:</h4>
              <ol className="list-decimal list-inside space-y-2 text-green-800 text-sm">
                <li><strong>Step 1 - Data Collection & Preprocessing:</strong> Fetch stock data, clean missing values, normalize features</li>
                <li><strong>Step 2 - Technical Indicators:</strong> Calculate SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator</li>
                <li><strong>Step 3 - Feature Engineering:</strong> Create sequences for time series, split train/val/test sets</li>
                <li><strong>Step 4 - LSTM Price Prediction:</strong> Build and train LSTM neural network for price forecasting</li>
                <li><strong>Step 5 - RL Trading Strategy:</strong> Implement DQN agent to learn optimal trading actions</li>
                <li><strong>Step 6 - Risk Management:</strong> Position sizing, stop-loss, performance metrics (Sharpe, Sortino, Max Drawdown)</li>
                <li><strong>Step 7 - Complete Integration:</strong> Full trading system combining all components</li>
              </ol>
            </div>
            
            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Formulas:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 space-y-4">
              
              {/* Simple Moving Average */}
              <div>
                <strong className="text-gray-800 text-base">1. Simple Moving Average (SMA):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">SMA(n) = (P₁ + P₂ + ... + Pₙ) / n</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>SMA(n)</strong>: Simple Moving Average over n periods</li>
                    <li><strong>Pᵢ</strong>: Price at time i</li>
                    <li><strong>n</strong>: Window size (number of periods)</li>
                    <li><strong>/</strong>: Division (average calculation)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Prices over 5 days: P₁ = 100, P₂ = 102, P₃ = 101, P₄ = 103, P₅ = 105</li>
                    <li><strong>Step 1:</strong> Sum all prices</li>
                    <li className="ml-4">Sum = 100 + 102 + 101 + 103 + 105 = 511</li>
                    <li><strong>Step 2:</strong> Divide by number of periods (n = 5)</li>
                    <li className="ml-4">SMA(5) = 511 / 5 = 102.2</li>
                    <li>SMA smooths out price fluctuations to show trend</li>
                  </ul>
                </div>
              </div>

              {/* Exponential Moving Average */}
              <div>
                <strong className="text-gray-800 text-base">2. Exponential Moving Average (EMA):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">EMA(t) = α · P(t) + (1-α) · EMA(t-1)</p>
                <p className="text-gray-700 text-sm mt-1 font-mono">where α = 2/(n+1)</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>EMA(t)</strong>: Exponential Moving Average at time t</li>
                    <li><strong>α</strong>: Smoothing factor (weight for current price)</li>
                    <li><strong>P(t)</strong>: Current price</li>
                    <li><strong>EMA(t-1)</strong>: Previous EMA value</li>
                    <li><strong>n</strong>: Window size</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Window size: n = 12</li>
                    <li><strong>Step 1:</strong> Calculate smoothing factor</li>
                    <li className="ml-4">α = 2/(12+1) = 2/13 = 0.154</li>
                    <li><strong>Step 2:</strong> Initialize EMA (use SMA for first value)</li>
                    <li className="ml-4">EMA(0) = 100 (first price)</li>
                    <li><strong>Step 3:</strong> Calculate EMA for next period</li>
                    <li className="ml-4">If P(1) = 102, EMA(0) = 100</li>
                    <li className="ml-4">EMA(1) = 0.154 × 102 + (1-0.154) × 100</li>
                    <li className="ml-4">EMA(1) = 15.708 + 84.6 = 100.308</li>
                    <li>EMA gives more weight to recent prices than SMA</li>
                  </ul>
                </div>
              </div>

              {/* RSI */}
              <div>
                <strong className="text-gray-800 text-base">3. Relative Strength Index (RSI):</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">RSI = 100 - (100 / (1 + RS))</p>
                <p className="text-gray-700 text-sm mt-1 font-mono">where RS = Average Gain / Average Loss</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>RSI</strong>: Relative Strength Index (0-100)</li>
                    <li><strong>RS</strong>: Relative Strength (ratio of gains to losses)</li>
                    <li><strong>Average Gain</strong>: Average of positive price changes over n periods</li>
                    <li><strong>Average Loss</strong>: Average of negative price changes over n periods</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>14-day period: Gains = [+2, +1, +3, +1, +2] (average = 1.8)</li>
                    <li>Losses = [-1, -2, -1] (average = -1.33, use absolute: 1.33)</li>
                    <li><strong>Step 1:</strong> Calculate RS</li>
                    <li className="ml-4">RS = Average Gain / Average Loss = 1.8 / 1.33 = 1.35</li>
                    <li><strong>Step 2:</strong> Calculate RSI</li>
                    <li className="ml-4">RSI = 100 - (100 / (1 + 1.35))</li>
                    <li className="ml-4">RSI = 100 - (100 / 2.35)</li>
                    <li className="ml-4">RSI = 100 - 42.55 = 57.45</li>
                    <li>RSI &gt; 70: Overbought (potential sell signal)</li>
                    <li>RSI &lt; 30: Oversold (potential buy signal)</li>
                  </ul>
                </div>
              </div>

              {/* Sharpe Ratio */}
              <div>
                <strong className="text-gray-800 text-base">4. Sharpe Ratio:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Sharpe = (Mean Return - Risk-Free Rate) / Std(Returns) × √252</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Sharpe</strong>: Risk-adjusted return metric</li>
                    <li><strong>Mean Return</strong>: Average daily return</li>
                    <li><strong>Risk-Free Rate</strong>: Risk-free interest rate (e.g., 2% annually)</li>
                    <li><strong>Std(Returns)</strong>: Standard deviation of returns (volatility)</li>
                    <li><strong>√252</strong>: Annualization factor (252 trading days per year)</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Daily returns: [0.01, 0.02, -0.01, 0.015, 0.005]</li>
                    <li>Risk-free rate: 2% annually = 0.02/252 = 0.000079 daily</li>
                    <li><strong>Step 1:</strong> Calculate mean return</li>
                    <li className="ml-4">Mean Return = (0.01 + 0.02 - 0.01 + 0.015 + 0.005) / 5 = 0.01</li>
                    <li><strong>Step 2:</strong> Calculate standard deviation</li>
                    <li className="ml-4">Std = 0.008 (example value)</li>
                    <li><strong>Step 3:</strong> Calculate Sharpe Ratio</li>
                    <li className="ml-4">Excess Return = 0.01 - 0.000079 = 0.009921</li>
                    <li className="ml-4">Sharpe = (0.009921 / 0.008) × √252</li>
                    <li className="ml-4">Sharpe = 1.24 × 15.87 = 19.68</li>
                    <li>Higher Sharpe = Better risk-adjusted returns</li>
                    <li>Sharpe &gt; 1: Good, &gt; 2: Very good, &gt; 3: Excellent</li>
                  </ul>
                </div>
              </div>

              {/* Q-Learning Update */}
              <div>
                <strong className="text-gray-800 text-base">5. Q-Learning Update:</strong>
                <p className="text-gray-700 text-sm mt-1 font-mono">Q(s, a) ← Q(s, a) + α[r + γ·max Q(s', a') - Q(s, a)]</p>
                <div className="mt-2 text-sm text-gray-600 space-y-1">
                  <p><strong>Components:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li><strong>Q(s, a)</strong>: Action-value function (expected future reward)</li>
                    <li><strong>α</strong>: Learning rate (step size, e.g., 0.001)</li>
                    <li><strong>r</strong>: Immediate reward from taking action a in state s</li>
                    <li><strong>γ</strong>: Discount factor (future reward importance, e.g., 0.95)</li>
                    <li><strong>max Q(s', a')</strong>: Maximum Q-value in next state s'</li>
                    <li><strong>←</strong>: Update assignment</li>
                  </ul>
                  <p className="mt-2"><strong>Example with Numbers:</strong></p>
                  <ul className="list-disc list-inside ml-2 space-y-1">
                    <li>Current state s: Stock price = 100, RSI = 50, Portfolio value = 10000</li>
                    <li>Action a: Buy</li>
                    <li>Current Q(s, a) = 0.5</li>
                    <li>Immediate reward r = 0.02 (2% gain)</li>
                    <li>Learning rate α = 0.1, Discount factor γ = 0.95</li>
                    <li>Next state s': Stock price = 102, RSI = 55, Portfolio value = 10200</li>
                    <li>Max Q(s', a') = max(Q(s', Buy), Q(s', Sell), Q(s', Hold)) = 0.8</li>
                    <li><strong>Step 1:</strong> Calculate target Q-value</li>
                    <li className="ml-4">Target = r + γ·max Q(s', a') = 0.02 + 0.95 × 0.8 = 0.02 + 0.76 = 0.78</li>
                    <li><strong>Step 2:</strong> Calculate error</li>
                    <li className="ml-4">Error = Target - Q(s, a) = 0.78 - 0.5 = 0.28</li>
                    <li><strong>Step 3:</strong> Update Q-value</li>
                    <li className="ml-4">Q(s, a) ← 0.5 + 0.1 × 0.28 = 0.5 + 0.028 = 0.528</li>
                    <li>Q-value increases, agent learns that buying in this state is beneficial</li>
                  </ul>
                </div>
              </div>

            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Concepts Used:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Time Series Analysis:</strong> Sequential data processing, trend identification</li>
              <li><strong>Technical Indicators:</strong> Mathematical formulas for market analysis (SMA, EMA, RSI, MACD)</li>
              <li><strong>LSTM Neural Networks:</strong> Long Short-Term Memory for sequence prediction</li>
              <li><strong>Reinforcement Learning:</strong> Q-Learning, DQN for strategy optimization</li>
              <li><strong>Risk Management:</strong> Position sizing, stop-loss, performance metrics</li>
              <li><strong>Statistics:</strong> Sharpe ratio, Sortino ratio, maximum drawdown</li>
              <li><strong>Backtesting:</strong> Historical strategy evaluation</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Real-World Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Algorithmic trading systems</li>
              <li>Portfolio optimization</li>
              <li>Risk management for financial institutions</li>
              <li>Automated trading bots</li>
              <li>Market prediction and forecasting</li>
              <li>Cryptocurrency trading</li>
              <li>Forex trading strategies</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

