import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function SupervisedLearningEducationalPanels({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'key-concepts' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Key Concepts & Terms in Supervised Learning</h3>
            <p className="text-gray-700 mb-4">
              Understanding fundamental supervised learning concepts is essential for building effective ML models. 
              This section explains regression, classification, and Support Vector Machines.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Regression vs Classification</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4 border-2 border-gray-200">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-blue-50 rounded p-3 border-2 border-blue-200">
                  <h5 className="font-bold text-blue-900 mb-2">Regression</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Predicts continuous values</li>
                    <li>• Output: Real numbers</li>
                    <li>• Examples: Price, temperature, age</li>
                    <li>• Loss: MSE, MAE</li>
                    <li>• Metrics: RMSE, R²</li>
                  </ul>
                </div>
                <div className="bg-purple-50 rounded p-3 border-2 border-purple-200">
                  <h5 className="font-bold text-purple-900 mb-2">Classification</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Predicts discrete categories</li>
                    <li>• Output: Class labels</li>
                    <li>• Examples: Spam/not spam, cat/dog</li>
                    <li>• Loss: Cross-entropy, Hinge</li>
                    <li>• Metrics: Accuracy, F1, ROC-AUC</li>
                  </ul>
                </div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Support Vector Machines (SVM)</h4>
            <p className="text-gray-700 mb-3">
              SVM is a powerful classification algorithm that finds the optimal decision boundary by maximizing 
              the margin between classes. It uses support vectors (critical data points) to define the boundary.
            </p>
            <div className="bg-orange-50 rounded-lg p-4 mb-4 border-2 border-orange-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Key Features:</strong></div>
                <div>• Maximizes margin between classes</div>
                <div>• Uses kernel trick for non-linear data</div>
                <div>• Memory efficient (only stores support vectors)</div>
                <div>• Loss function: Hinge Loss</div>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="key-concepts" />
        </div>
      )}

      {selectedTopic === 'foundations' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Mathematical Foundations of Supervised Learning</h3>
            <p className="text-gray-700 mb-4">
              Supervised learning maps input → output using labeled data. The math behind it primarily involves 
              <strong> Linear Algebra</strong>, <strong>Calculus</strong>, <strong>Probability</strong>, and <strong>Optimization</strong>.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">A. Linear Algebra</h4>
            <p className="text-gray-700 mb-3">
              Used to represent datasets and model parameters as matrices/vectors.
            </p>
            <div className="bg-indigo-50 rounded-lg p-4 mb-4 border-2 border-indigo-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Dataset X</strong> → matrix of size <strong>m × n</strong> (m = samples, n = features)</div>
                <div><strong>Parameters θ</strong> → vector of weights</div>
                <div className="mt-2"><strong>Prediction:</strong> <strong>ŷ = Xθ</strong></div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">B. Calculus (Differential Calculus)</h4>
            <p className="text-gray-700 mb-3">
              Used in training: adjusting weights to minimize loss (error).
            </p>
            <div className="bg-purple-50 rounded-lg p-4 mb-4 border-2 border-purple-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Gradient Descent:</strong> θ := θ - α × (∂J(θ) / ∂θ)</div>
                <div className="mt-2"><strong>Loss for Linear Regression:</strong></div>
                <div>J(θ) = (1/2m) × Σ(hθ(xᵢ) - yᵢ)²</div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">C. Probability and Statistics</h4>
            <p className="text-gray-700 mb-3">
              Helps model uncertainty and interpret predictions.
            </p>
            <div className="bg-blue-50 rounded-lg p-4 mb-4 border-2 border-blue-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Logistic Regression:</strong> P(y = 1 | x) = 1 / (1 + e^(-θᵀx))</div>
                <div className="mt-2"><strong>Naive Bayes:</strong> Uses Bayes' theorem for classification</div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">D. Optimization</h4>
            <p className="text-gray-700 mb-3">
              Finding the best weights θ using techniques like gradient descent or stochastic gradient descent (SGD).
            </p>
            <div className="bg-orange-50 rounded-lg p-4 mb-4 border-2 border-orange-200">
              <div className="font-mono text-sm space-y-2">
                <div><strong>Gradient Descent:</strong> θ := θ - α × (∂J(θ) / ∂θ)</div>
                <div className="mt-2"><strong>Stochastic Gradient Descent (SGD):</strong> Uses mini-batches for faster updates</div>
                <div className="mt-2"><strong>Convex Optimization:</strong> Ensures global minimum for convex loss functions (e.g., linear regression)</div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">How They Work Together:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li><strong>Linear Algebra</strong> represents the data and model structure</li>
              <li><strong>Calculus</strong> computes gradients to guide optimization</li>
              <li><strong>Optimization</strong> finds the best weights using gradient descent or SGD</li>
              <li><strong>Probability</strong> provides probabilistic interpretations and handles uncertainty</li>
              <li>Together, they enable machines to learn from labeled examples</li>
            </ol>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="foundations" />
        </div>
      )}

      {selectedTopic === 'loss-functions' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Loss Functions in Machine Learning</h3>
            <p className="text-gray-700 mb-4">
              Loss functions measure how far our model's predictions are from the actual values. 
              They guide the optimization process by quantifying prediction error.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Common Loss Functions:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Mean Squared Error (MSE):</strong> Used for regression. Penalizes large errors more than small ones.</li>
              <li><strong>Mean Absolute Error (MAE):</strong> Used for regression. Linear penalty, robust to outliers.</li>
              <li><strong>Cross-Entropy:</strong> Used for classification. Measures difference between predicted and true probability distributions.</li>
              <li><strong>Hinge Loss:</strong> Used for Support Vector Machines. Encourages correct classification with margin.</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Key Properties:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Loss functions are differentiable (except some like MAE at zero)</li>
              <li>They guide gradient descent optimization</li>
              <li>Choice of loss function depends on problem type (regression vs classification)</li>
              <li>Some loss functions are derived from probability distributions (MSE from Normal, Cross-entropy from Bernoulli)</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="loss-functions" />
        </div>
      )}

      {selectedTopic === 'model-evaluation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Model Evaluation Metrics</h3>
            <p className="text-gray-700 mb-4">
              Evaluation metrics help us assess how well our model performs on unseen data. 
              Different metrics are appropriate for different types of problems.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Classification Metrics:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Accuracy:</strong> Overall correctness (TP + TN) / Total</li>
              <li><strong>Precision:</strong> Of predicted positives, how many are correct? TP / (TP + FP)</li>
              <li><strong>Recall (Sensitivity):</strong> Of actual positives, how many did we catch? TP / (TP + FN)</li>
              <li><strong>F1-Score:</strong> Harmonic mean of precision and recall. Balances both metrics.</li>
              <li><strong>ROC-AUC:</strong> Area under ROC curve. Measures overall classifier performance.</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Confusion Matrix:</h4>
            <p className="text-gray-700 mb-4">
              A confusion matrix shows the breakdown of predictions vs actual values:
            </p>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div></div>
                <div className="text-center font-semibold">Predicted: -</div>
                <div className="text-center font-semibold">Predicted: +</div>
                <div className="text-center font-semibold">Actual: -</div>
                <div className="bg-green-100 p-2 rounded text-center">TN</div>
                <div className="bg-yellow-100 p-2 rounded text-center">FP</div>
                <div className="text-center font-semibold">Actual: +</div>
                <div className="bg-red-100 p-2 rounded text-center">FN</div>
                <div className="bg-blue-100 p-2 rounded text-center">TP</div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">ROC Curve:</h4>
            <p className="text-gray-700 mb-4">
              The ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate 
              at different classification thresholds. AUC (Area Under Curve) summarizes overall performance.
            </p>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="model-evaluation" />
        </div>
      )}

      {selectedTopic === 'bias-variance' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Bias-Variance Tradeoff</h3>
            <p className="text-gray-700 mb-4">
              The bias-variance tradeoff is a fundamental concept in machine learning that describes 
              the relationship between model complexity and generalization error.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Understanding Bias and Variance:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Bias:</strong> Error from oversimplifying assumptions. High bias = model misses relevant patterns (underfitting)</li>
              <li><strong>Variance:</strong> Error from sensitivity to small fluctuations. High variance = model fits noise (overfitting)</li>
              <li><strong>Total Error:</strong> Bias² + Variance + Irreducible Error</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">The Tradeoff:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>As model complexity increases, bias decreases but variance increases</li>
              <li>Simple models: High bias, low variance (underfitting)</li>
              <li>Complex models: Low bias, high variance (overfitting)</li>
              <li>Goal: Find the sweet spot that minimizes total error</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Solutions:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Use validation set to monitor performance</li>
              <li>Cross-validation to get better estimates</li>
              <li>Regularization to control complexity</li>
              <li>Early stopping to prevent overfitting</li>
              <li>Ensemble methods to reduce variance</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="bias-variance" />
        </div>
      )}

      {selectedTopic === 'regularization' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-green-900 mb-3">Regularization</h3>
            <p className="text-gray-700 mb-4">
              Regularization is a technique to prevent overfitting by adding a penalty term to the loss function. 
              It encourages simpler models that generalize better to unseen data.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">L1 Regularization (Lasso):</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Penalty: λ × Σ|w| (sum of absolute values of weights)</li>
              <li>Effect: Shrinks weights towards zero, can set weights to exactly zero</li>
              <li>Use case: Feature selection - removes irrelevant features</li>
              <li>Mathematical property: Creates sparse solutions</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">L2 Regularization (Ridge):</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Penalty: λ × Σw² (sum of squared values of weights)</li>
              <li>Effect: Shrinks weights proportionally, keeps all features</li>
              <li>Use case: When all features might be relevant but you want to prevent overfitting</li>
              <li>Mathematical property: Smooth shrinkage, no feature elimination</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Elastic Net:</h4>
            <p className="text-gray-700 mb-4">
              Combines L1 and L2 regularization: λ₁ × Σ|w| + λ₂ × Σw². 
              Provides benefits of both: feature selection (L1) and smooth shrinkage (L2).
            </p>

            <h4 className="font-semibold text-gray-800 mb-2">Choosing λ (Lambda):</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>λ = 0: No regularization (original model)</li>
              <li>Small λ: Weak regularization, model stays complex</li>
              <li>Large λ: Strong regularization, simpler model</li>
              <li>Use cross-validation to find optimal λ</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="regularization" />
        </div>
      )}
    </div>
  );
}

