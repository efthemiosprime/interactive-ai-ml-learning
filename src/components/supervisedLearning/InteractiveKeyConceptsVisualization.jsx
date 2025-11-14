import React, { useState } from 'react';
import { TrendingUp, Target, Zap, Brain } from 'lucide-react';

export default function InteractiveKeyConceptsVisualization() {
  const [selectedConcept, setSelectedConcept] = useState('regression');

  return (
    <div className="space-y-6">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Explore key supervised learning concepts and understand the difference between regression and classification!
        </p>
      </div>

      {/* Concept Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Concept
        </label>
        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => setSelectedConcept('regression')}
            className={`px-4 py-3 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
              selectedConcept === 'regression'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            Regression
          </button>
          <button
            onClick={() => setSelectedConcept('classification')}
            className={`px-4 py-3 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
              selectedConcept === 'classification'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Target className="w-4 h-4" />
            Classification
          </button>
          <button
            onClick={() => setSelectedConcept('svm')}
            className={`px-4 py-3 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 ${
              selectedConcept === 'svm'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Zap className="w-4 h-4" />
            SVM
          </button>
        </div>
      </div>

      {/* Regression */}
      {selectedConcept === 'regression' && (
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-200">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-8 h-8 text-blue-600" />
            <h4 className="text-2xl font-bold text-blue-900">Regression</h4>
          </div>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Definition</h5>
              <p className="text-gray-700 mb-3">
                <strong>Regression</strong> is a supervised learning task where the goal is to predict a continuous numerical value.
                The output is a real number (e.g., price, temperature, age, salary).
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Key Characteristics</h5>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li><strong>Output:</strong> Continuous values (real numbers)</li>
                <li><strong>Examples:</strong> House prices, stock prices, temperature, height, weight</li>
                <li><strong>Loss Function:</strong> Mean Squared Error (MSE) or Mean Absolute Error (MAE)</li>
                <li><strong>Evaluation Metrics:</strong> RMSE, MAE, R² (coefficient of determination)</li>
              </ul>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Mathematical Form</h5>
              <div className="bg-blue-50 rounded p-3 mb-2">
                <div className="font-mono text-sm space-y-2">
                  <div><strong>Linear Regression:</strong> y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ</div>
                  <div className="text-xs text-gray-600 mt-2">
                    Where y is the predicted continuous value, xᵢ are features, and θᵢ are weights
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Common Algorithms</h5>
              <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700">
                <div>
                  <strong>Linear Regression:</strong> Fits a straight line through data points
                </div>
                <div>
                  <strong>Polynomial Regression:</strong> Fits a polynomial curve
                </div>
                <div>
                  <strong>Ridge Regression:</strong> Linear regression with L2 regularization
                </div>
                <div>
                  <strong>Lasso Regression:</strong> Linear regression with L1 regularization
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Real-World Examples</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li><strong>House Price Prediction:</strong> Predict price based on size, location, bedrooms</li>
                <li><strong>Stock Price Forecasting:</strong> Predict future stock prices</li>
                <li><strong>Weather Prediction:</strong> Predict temperature, rainfall</li>
                <li><strong>Sales Forecasting:</strong> Predict future sales based on historical data</li>
              </ul>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Regression predicts "how much" or "how many" - continuous numerical values. 
                The model learns a function that maps input features to a continuous output value.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Classification */}
      {selectedConcept === 'classification' && (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-200">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-8 h-8 text-purple-600" />
            <h4 className="text-2xl font-bold text-purple-900">Classification</h4>
          </div>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Definition</h5>
              <p className="text-gray-700 mb-3">
                <strong>Classification</strong> is a supervised learning task where the goal is to predict a discrete category or class label.
                The output is a class (e.g., spam/not spam, cat/dog/bird, positive/negative).
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Types of Classification</h5>
              <div className="space-y-3">
                <div className="bg-purple-50 rounded p-3">
                  <strong className="text-purple-900">Binary Classification:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    Two classes (e.g., spam/not spam, fraud/legitimate, positive/negative)
                  </p>
                </div>
                <div className="bg-purple-50 rounded p-3">
                  <strong className="text-purple-900">Multi-class Classification:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    More than two classes (e.g., cat/dog/bird, sentiment: positive/neutral/negative)
                  </p>
                </div>
                <div className="bg-purple-50 rounded p-3">
                  <strong className="text-purple-900">Multi-label Classification:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    Multiple labels per instance (e.g., image tags: beach, sunset, ocean)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Key Characteristics</h5>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li><strong>Output:</strong> Discrete categories/classes</li>
                <li><strong>Examples:</strong> Email spam detection, image recognition, medical diagnosis</li>
                <li><strong>Loss Function:</strong> Cross-Entropy Loss, Hinge Loss</li>
                <li><strong>Evaluation Metrics:</strong> Accuracy, Precision, Recall, F1-Score, ROC-AUC</li>
              </ul>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Mathematical Form</h5>
              <div className="bg-purple-50 rounded p-3 mb-2">
                <div className="font-mono text-sm space-y-2">
                  <div><strong>Logistic Regression:</strong> P(y = 1 | x) = 1 / (1 + e^(-θᵀx))</div>
                  <div className="text-xs text-gray-600 mt-2">
                    Outputs probability of belonging to a class, then thresholded to make class prediction
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Common Algorithms</h5>
              <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700">
                <div>
                  <strong>Logistic Regression:</strong> Binary/multi-class classification using sigmoid/softmax
                </div>
                <div>
                  <strong>Decision Trees:</strong> Tree-based classification rules
                </div>
                <div>
                  <strong>Random Forest:</strong> Ensemble of decision trees
                </div>
                <div>
                  <strong>Naive Bayes:</strong> Probabilistic classifier using Bayes' theorem
                </div>
                <div>
                  <strong>Support Vector Machines (SVM):</strong> Finds optimal decision boundary
                </div>
                <div>
                  <strong>Neural Networks:</strong> Deep learning for complex classification
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Real-World Examples</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li><strong>Email Spam Detection:</strong> Classify emails as spam or not spam</li>
                <li><strong>Image Recognition:</strong> Classify images (cat, dog, bird, etc.)</li>
                <li><strong>Medical Diagnosis:</strong> Classify diseases based on symptoms</li>
                <li><strong>Sentiment Analysis:</strong> Classify text as positive/negative/neutral</li>
                <li><strong>Fraud Detection:</strong> Classify transactions as fraudulent or legitimate</li>
              </ul>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Classification predicts "which category" or "what class" - discrete labels. 
                The model learns to separate data into distinct categories and assigns each input to one or more classes.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Support Vector Machines */}
      {selectedConcept === 'svm' && (
        <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg p-6 border-2 border-orange-200">
          <div className="flex items-center gap-3 mb-4">
            <Zap className="w-8 h-8 text-orange-600" />
            <h4 className="text-2xl font-bold text-orange-900">Support Vector Machines (SVM)</h4>
          </div>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Definition</h5>
              <p className="text-gray-700 mb-3">
                <strong>Support Vector Machines (SVM)</strong> is a powerful classification algorithm that finds the optimal 
                decision boundary (hyperplane) that maximizes the margin between different classes. 
                It can handle both linear and non-linear classification problems.
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Key Concepts</h5>
              <div className="space-y-3">
                <div className="bg-orange-50 rounded p-3">
                  <strong className="text-orange-900">Support Vectors:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    The data points closest to the decision boundary. These are the "critical" points that define the margin.
                  </p>
                </div>
                <div className="bg-orange-50 rounded p-3">
                  <strong className="text-orange-900">Margin:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    The distance between the decision boundary and the nearest data points from each class. 
                    SVM maximizes this margin.
                  </p>
                </div>
                <div className="bg-orange-50 rounded p-3">
                  <strong className="text-orange-900">Hyperplane:</strong>
                  <p className="text-sm text-gray-700 mt-1">
                    The decision boundary that separates classes. In 2D, it's a line; in 3D, it's a plane; 
                    in higher dimensions, it's a hyperplane.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Mathematical Formulation</h5>
              <div className="bg-orange-50 rounded p-3 mb-2">
                <div className="font-mono text-sm space-y-2">
                  <div><strong>Decision Function:</strong> f(x) = sign(wᵀx + b)</div>
                  <div className="text-xs text-gray-600 mt-2">
                    Where w is the weight vector, b is the bias, and sign() returns +1 or -1
                  </div>
                  <div className="mt-3"><strong>Optimization Objective:</strong></div>
                  <div>Minimize: (1/2)||w||² + C × Σξᵢ</div>
                  <div className="text-xs text-gray-600 mt-2">
                    Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Key Features</h5>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li><strong>Kernel Trick:</strong> Can handle non-linear data using kernel functions (RBF, polynomial, sigmoid)</li>
                <li><strong>Margin Maximization:</strong> Finds the widest possible margin between classes</li>
                <li><strong>Robust:</strong> Works well with high-dimensional data</li>
                <li><strong>Memory Efficient:</strong> Only uses support vectors for prediction (not all training data)</li>
                <li><strong>Regularization:</strong> C parameter controls the tradeoff between margin size and classification errors</li>
              </ul>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Types of SVM</h5>
              <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700">
                <div>
                  <strong>Hard Margin SVM:</strong> No misclassifications allowed (C = ∞)
                </div>
                <div>
                  <strong>Soft Margin SVM:</strong> Allows some misclassifications (C &lt; ∞)
                </div>
                <div>
                  <strong>Linear SVM:</strong> Uses linear kernel for linearly separable data
                </div>
                <div>
                  <strong>Non-linear SVM:</strong> Uses kernel functions (RBF, polynomial) for non-linear data
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Loss Function</h5>
              <div className="bg-orange-50 rounded p-3 mb-2">
                <div className="font-mono text-sm">
                  <strong>Hinge Loss:</strong> L(y, f(x)) = max(0, 1 - y × f(x))
                </div>
                <div className="text-xs text-gray-600 mt-2">
                  Penalizes predictions that are on the wrong side of the margin or within the margin
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Real-World Applications</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li><strong>Text Classification:</strong> Spam detection, sentiment analysis</li>
                <li><strong>Image Classification:</strong> Handwritten digit recognition, face detection</li>
                <li><strong>Bioinformatics:</strong> Protein classification, gene expression analysis</li>
                <li><strong>Medical Diagnosis:</strong> Disease classification based on symptoms</li>
                <li><strong>Financial Analysis:</strong> Credit scoring, fraud detection</li>
              </ul>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Advantages & Disadvantages</h5>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <strong className="text-green-700">Advantages:</strong>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 mt-1">
                    <li>Effective in high-dimensional spaces</li>
                    <li>Memory efficient (uses support vectors only)</li>
                    <li>Versatile (different kernel functions)</li>
                    <li>Works well with small datasets</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-red-700">Disadvantages:</strong>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 mt-1">
                    <li>Doesn't perform well with large datasets</li>
                    <li>Doesn't provide probability estimates directly</li>
                    <li>Sensitive to feature scaling</li>
                    <li>Kernel selection can be tricky</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                SVM finds the optimal decision boundary by maximizing the margin between classes. 
                It's particularly powerful for high-dimensional data and can handle non-linear relationships 
                through the kernel trick. The algorithm focuses on the "hard" examples (support vectors) 
                near the decision boundary, making it robust and efficient.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

