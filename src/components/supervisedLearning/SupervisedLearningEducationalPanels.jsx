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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Key Concepts in Supervised Learning?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Understanding key concepts helps us <strong className="text-green-700">choose the right approach</strong> 
                for each problem. Regression vs classification, and algorithms like SVM, determine 
                which tools to use and how to structure our ML solutions.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Problem Formulation</h4>
                    <p className="text-gray-700">
                      Choosing between regression and classification <strong>determines everything</strong> - 
                      which loss function to use, which metrics to evaluate, and which algorithms to consider. 
                      This fundamental decision shapes the entire ML pipeline.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> House price prediction = regression (MSE loss, RMSE metric). 
                      Spam detection = classification (cross-entropy loss, F1 metric). 
                      Wrong choice = wrong approach = poor results.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔧</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Algorithm Selection</h4>
                    <p className="text-gray-700">
                      Different algorithms suit different problems. SVM excels with <strong>high-dimensional sparse data</strong> 
                      (like text), while decision trees work well with interpretable features. 
                      Understanding these concepts guides algorithm selection.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Text classification (thousands of word features) → use SVM. 
                      Medical diagnosis (few interpretable features) → use decision trees for interpretability.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📈</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Performance Optimization</h4>
                    <p className="text-gray-700">
                      Understanding concepts like margin maximization (SVM) helps us <strong>optimize model performance</strong>. 
                      We can tune hyperparameters, select features, and design architectures 
                      based on these fundamental principles.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Knowing SVM maximizes margin helps us choose the right C parameter 
                      and kernel function for optimal performance.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Key concepts are the <strong>vocabulary and grammar of ML</strong>. They help us communicate 
                  about problems, choose appropriate solutions, and understand why certain approaches work. 
                  Without understanding these fundamentals, we're just applying algorithms blindly 
                  without knowing why or when they're appropriate.
                </p>
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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Mathematical Foundations?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Mathematical foundations provide the <strong className="text-green-700">theoretical framework</strong> 
                that makes supervised learning possible. They enable us to represent data, optimize models, 
                and reason about predictions systematically.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Efficient Data Representation</h4>
                    <p className="text-gray-700">
                      Linear algebra allows us to represent <strong>entire datasets as matrices</strong>, 
                      enabling efficient computation. Instead of processing samples one by one, 
                      we can process thousands simultaneously using matrix operations.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Matrix multiplication Xθ computes predictions for all 10,000 samples 
                      in one operation, not 10,000 separate operations.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚡</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Systematic Optimization</h4>
                    <p className="text-gray-700">
                      Calculus provides <strong>gradients that guide optimization</strong>. 
                      Without derivatives, we'd have to try random parameter values. 
                      Gradients tell us exactly how to adjust parameters to improve performance.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Gradient descent converges to optimal parameters in hundreds of iterations. 
                      Random search might take millions of trials.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎲</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Uncertainty Quantification</h4>
                    <p className="text-gray-700">
                      Probability theory enables us to <strong>quantify uncertainty</strong> in predictions. 
                      Instead of just predicting a value, we can predict a distribution, 
                      providing confidence intervals and uncertainty estimates.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Logistic regression outputs probabilities (0.7 = 70% confident), 
                      not just binary predictions (spam/not spam).
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Mathematical foundations transform supervised learning from "trial and error" into 
                  <strong> systematic, principled optimization</strong>. They enable efficient computation, 
                  guide optimization, and provide theoretical guarantees. Without these foundations, 
                  we couldn't build reliable, scalable ML systems.
                </p>
              </div>
            </div>
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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Loss Functions?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Loss functions provide the <strong className="text-green-700">objective for optimization</strong>. 
                They quantify how wrong our predictions are and guide the model to improve. 
                Without loss functions, we have no way to train models.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🧭</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Optimization Guidance</h4>
                    <p className="text-gray-700">
                      Loss functions tell the optimizer <strong>which direction to move</strong>. 
                      The gradient of the loss function points toward better parameters. 
                      Without a loss function, gradient descent has no direction to follow.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> MSE loss gradient tells us: "decrease weights here, increase weights there" 
                      to reduce prediction error.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Problem-Specific Optimization</h4>
                    <p className="text-gray-700">
                      Different loss functions optimize for <strong>different objectives</strong>. 
                      MSE penalizes large errors heavily (good for regression). 
                      Cross-entropy optimizes probability calibration (good for classification). 
                      Choosing the right loss aligns optimization with our goals.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Using MSE for classification would optimize the wrong thing - 
                      we want probability calibration, not squared error.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Convergence & Training</h4>
                    <p className="text-gray-700">
                      Loss functions provide a <strong>measurable objective</strong> that decreases during training. 
                      We can monitor loss to detect convergence, overfitting, or training issues. 
                      Loss curves are essential for debugging and understanding model behavior.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> If loss plateaus, we've converged. If validation loss increases while 
                      training loss decreases, we're overfitting.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Loss functions are the <strong>compass for machine learning</strong>. They define what "good" means 
                  and guide the model toward better performance. Without loss functions, we can't train models, 
                  can't optimize parameters, and can't measure progress. They're the foundation of all supervised learning.
                </p>
              </div>
            </div>
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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Model Evaluation Metrics?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Evaluation metrics provide <strong className="text-green-700">objective measures</strong> of model performance. 
                They help us compare models, detect issues, and make informed decisions about 
                which model to deploy in production.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Model Comparison & Selection</h4>
                    <p className="text-gray-700">
                      Metrics allow us to <strong>compare different models objectively</strong>. 
                      We can test multiple algorithms, hyperparameters, or architectures, 
                      and choose the best one based on metrics like accuracy, F1-score, or AUC.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Model A: 85% accuracy, Model B: 90% accuracy → choose Model B. 
                      But also check precision/recall to understand trade-offs.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🚨</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Detecting Model Issues</h4>
                    <p className="text-gray-700">
                      Metrics reveal <strong>specific problems</strong> with our models. 
                      Low precision means too many false positives. Low recall means missing too many positives. 
                      These insights guide model improvement.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> High precision but low recall → model is too conservative. 
                      Adjust threshold or improve feature engineering to catch more positives.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">✅</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Production Readiness</h4>
                    <p className="text-gray-700">
                      Metrics help determine if a model is <strong>ready for production</strong>. 
                      We set performance thresholds (e.g., F1 &gt; 0.9) and only deploy models that meet them. 
                      Metrics also help monitor model performance over time.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> If production model's accuracy drops from 90% to 75%, 
                      metrics alert us to retrain or investigate data drift.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Evaluation metrics are the <strong>quality assurance system for ML</strong>. 
                  They tell us if our models work, how well they work, and where they fail. 
                  Without metrics, we're deploying models blindly, with no way to know if they're 
                  actually solving the problem or just appearing to work.
                </p>
              </div>
            </div>
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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need to Understand Bias-Variance Tradeoff?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                The bias-variance tradeoff helps us <strong className="text-green-700">diagnose and fix model problems</strong>. 
                Understanding this concept enables us to identify whether a model is underfitting or overfitting, 
                and guides us toward solutions.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔍</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Problem Diagnosis</h4>
                    <p className="text-gray-700">
                      Bias-variance tradeoff helps us <strong>diagnose why models fail</strong>. 
                      High bias (underfitting) means the model is too simple. 
                      High variance (overfitting) means the model memorized training data. 
                      This diagnosis guides solution selection.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> If training and validation error are both high → high bias → 
                      increase model complexity. If training error is low but validation error is high → 
                      high variance → add regularization.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚖️</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Balancing Complexity</h4>
                    <p className="text-gray-700">
                      The tradeoff teaches us to <strong>balance model complexity</strong>. 
                      Too simple = underfitting (high bias). Too complex = overfitting (high variance). 
                      We need to find the sweet spot that minimizes total error.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> A polynomial of degree 1 might underfit (high bias). 
                      Degree 20 might overfit (high variance). Degree 3-5 might be optimal.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🛠️</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Solution Selection</h4>
                    <p className="text-gray-700">
                      Understanding bias-variance helps us <strong>choose the right solutions</strong>. 
                      High bias → increase complexity, add features, use more powerful models. 
                      High variance → add regularization, reduce features, use simpler models, get more data.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> If overfitting (high variance), don't add more layers - 
                      that makes it worse! Instead, add dropout or L2 regularization.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  The bias-variance tradeoff is the <strong>diagnostic framework for ML problems</strong>. 
                  It explains why models fail and guides us toward solutions. Without understanding this tradeoff, 
                  we might try random fixes (add more data, change optimizer) without knowing if they address 
                  the actual problem (bias vs variance).
                </p>
              </div>
            </div>
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

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Regularization?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Regularization prevents <strong className="text-green-700">overfitting</strong> by constraining model complexity. 
                It's essential for building models that generalize well to unseen data, 
                especially when we have limited training data or many features.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🛡️</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Preventing Overfitting</h4>
                    <p className="text-gray-700">
                      Regularization <strong>prevents models from memorizing training data</strong>. 
                      By penalizing large weights, it encourages simpler models that capture 
                      general patterns rather than noise. This improves generalization.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Without regularization, a model might achieve 99% training accuracy 
                      but only 70% test accuracy (overfitting). With regularization, 85% training and 85% test (generalizes).
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Feature Selection (L1)</h4>
                    <p className="text-gray-700">
                      L1 regularization performs <strong>automatic feature selection</strong> by setting 
                      irrelevant feature weights to zero. This reduces model complexity and improves 
                      interpretability by keeping only important features.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> With 1000 features, L1 might set 900 weights to zero, 
                      keeping only the 100 most important features.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚖️</div>
                  <div>
                    <h4 className="font-bold text-green-900 mb-2">Balancing Fit and Complexity</h4>
                    <p className="text-gray-700">
                      Regularization provides a <strong>tunable way to balance fit and complexity</strong>. 
                      By adjusting λ, we control how much we penalize complexity. 
                      This gives us fine-grained control over the bias-variance tradeoff.
                    </p>
                    <div className="mt-2 bg-green-50 rounded p-2 text-sm text-green-800">
                      <strong>Example:</strong> Small λ = complex model (low bias, high variance). 
                      Large λ = simple model (high bias, low variance). 
                      Optimal λ = balanced model (minimal total error).
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-100 to-emerald-100 rounded-lg p-4 border-2 border-green-300">
                <h4 className="font-bold text-green-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Regularization is the <strong>guardian against overfitting</strong>. It's especially crucial 
                  when we have many features relative to samples, or when using complex models. 
                  Without regularization, models often memorize training data and fail on new data. 
                  With regularization, we build models that generalize and perform well in production.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="supervised-learning" operationType="regularization" />
        </div>
      )}
    </div>
  );
}

