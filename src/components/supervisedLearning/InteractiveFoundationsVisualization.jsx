import React, { useState } from 'react';
import { Calculator, Brain, BarChart3, Target } from 'lucide-react';

export default function InteractiveFoundationsVisualization() {
  const [selectedFoundation, setSelectedFoundation] = useState('overview');

  return (
    <div className="space-y-6">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Explore how Linear Algebra, Calculus, and Probability form the foundation of supervised learning!
        </p>
      </div>

      {/* Foundation Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Mathematical Foundation
        </label>
        <div className="grid grid-cols-5 gap-2">
          <button
            onClick={() => setSelectedFoundation('overview')}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
              selectedFoundation === 'overview'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setSelectedFoundation('linear-algebra')}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1 ${
              selectedFoundation === 'linear-algebra'
                ? 'bg-indigo-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Calculator className="w-3 h-3" />
            Linear Algebra
          </button>
          <button
            onClick={() => setSelectedFoundation('calculus')}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1 ${
              selectedFoundation === 'calculus'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Brain className="w-3 h-3" />
            Calculus
          </button>
          <button
            onClick={() => setSelectedFoundation('probability')}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1 ${
              selectedFoundation === 'probability'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <BarChart3 className="w-3 h-3" />
            Probability
          </button>
          <button
            onClick={() => setSelectedFoundation('optimization')}
            className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1 ${
              selectedFoundation === 'optimization'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Target className="w-3 h-3" />
            Optimization
          </button>
        </div>
      </div>

      {/* Overview */}
      {selectedFoundation === 'overview' && (
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-6 border-2 border-green-200">
          <h4 className="text-xl font-bold text-green-900 mb-4">Supervised Learning: Input → Output</h4>
          <p className="text-gray-700 mb-4">
            Supervised learning maps input → output using labeled data. The math behind it primarily involves 
            <strong> Linear Algebra</strong>, <strong>Calculus</strong>, <strong>Probability</strong>, and <strong>Optimization</strong>.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
            <div className="bg-white rounded-lg p-4 border-2 border-indigo-200">
              <div className="flex items-center gap-2 mb-2">
                <Calculator className="w-6 h-6 text-indigo-600" />
                <h5 className="font-bold text-indigo-900">Linear Algebra</h5>
              </div>
              <p className="text-sm text-gray-700">
                Represents datasets and model parameters as matrices/vectors. Used for predictions and transformations.
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-6 h-6 text-purple-600" />
                <h5 className="font-bold text-purple-900">Calculus</h5>
              </div>
              <p className="text-sm text-gray-700">
                Used in training: adjusting weights to minimize loss. Gradient descent computes derivatives.
              </p>
            </div>
            
            <div className="bg-white rounded-lg p-4 border-2 border-blue-200">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-6 h-6 text-blue-600" />
                <h5 className="font-bold text-blue-900">Probability</h5>
              </div>
              <p className="text-sm text-gray-700">
                Models uncertainty and interprets predictions. Used in logistic regression and Naive Bayes.
              </p>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-200">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-6 h-6 text-orange-600" />
                <h5 className="font-bold text-orange-900">Optimization</h5>
              </div>
              <p className="text-sm text-gray-700">
                Finding the best weights θ using gradient descent or SGD. Convex optimization ensures global minimum.
              </p>
            </div>
          </div>

          {/* Visual Flow */}
          <div className="mt-6 bg-white rounded-lg p-4 border-2 border-green-300">
            <h5 className="font-bold text-green-900 mb-3">The Learning Process:</h5>
            <div className="flex items-center justify-between text-sm flex-wrap gap-2">
              <div className="text-center flex-1 min-w-[120px]">
                <div className="bg-indigo-100 rounded-lg p-3 mb-2">
                  <strong>1. Linear Algebra</strong><br />
                  <span className="text-xs">Represent data & parameters</span>
                </div>
              </div>
              <div className="text-2xl text-green-600">→</div>
              <div className="text-center flex-1 min-w-[120px]">
                <div className="bg-purple-100 rounded-lg p-3 mb-2">
                  <strong>2. Calculus</strong><br />
                  <span className="text-xs">Compute gradients</span>
                </div>
              </div>
              <div className="text-2xl text-green-600">→</div>
              <div className="text-center flex-1 min-w-[120px]">
                <div className="bg-orange-100 rounded-lg p-3 mb-2">
                  <strong>3. Optimization</strong><br />
                  <span className="text-xs">Find best weights</span>
                </div>
              </div>
              <div className="text-2xl text-green-600">→</div>
              <div className="text-center flex-1 min-w-[120px]">
                <div className="bg-blue-100 rounded-lg p-3 mb-2">
                  <strong>4. Probability</strong><br />
                  <span className="text-xs">Interpret predictions</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Linear Algebra */}
      {selectedFoundation === 'linear-algebra' && (
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
          <h4 className="text-xl font-bold text-indigo-900 mb-4">Linear Algebra in Supervised Learning</h4>
          <p className="text-gray-700 mb-4">
            Used to represent datasets and model parameters as matrices/vectors.
          </p>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-indigo-300">
              <h5 className="font-bold text-indigo-900 mb-2">Dataset Representation</h5>
              <div className="bg-indigo-50 rounded p-3 mb-2">
                <div className="font-mono text-sm">
                  <div className="mb-2">Dataset <strong>X</strong> → matrix of size <strong>m × n</strong></div>
                  <div className="text-xs text-gray-600">
                    • m = number of samples<br />
                    • n = number of features
                  </div>
                </div>
              </div>
              <div className="bg-gray-100 rounded p-2 font-mono text-xs">
                X = [x₁, x₂, ..., xₙ] where each xᵢ is a feature vector
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-indigo-300">
              <h5 className="font-bold text-indigo-900 mb-2">Parameters Representation</h5>
              <div className="bg-indigo-50 rounded p-3 mb-2">
                <div className="font-mono text-sm">
                  Parameters <strong>θ</strong> → vector of weights
                </div>
              </div>
              <div className="bg-gray-100 rounded p-2 font-mono text-xs">
                θ = [θ₀, θ₁, θ₂, ..., θₙ]ᵀ
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-indigo-300">
              <h5 className="font-bold text-indigo-900 mb-2">Prediction Formula</h5>
              <div className="bg-indigo-50 rounded p-3 mb-2">
                <div className="font-mono text-lg text-center mb-2">
                  <strong>ŷ = Xθ</strong>
                </div>
                <div className="text-xs text-gray-600 text-center">
                  Matrix multiplication: predictions = data × parameters
                </div>
              </div>
              <div className="bg-gray-100 rounded p-2 font-mono text-xs space-y-1">
                <div>For linear regression: ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ</div>
                <div>In matrix form: ŷ = [1, x₁, x₂, ..., xₙ] × [θ₀, θ₁, θ₂, ..., θₙ]ᵀ</div>
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Linear algebra enables efficient computation of predictions for all samples simultaneously using matrix operations, 
                which is much faster than computing predictions one by one.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Calculus */}
      {selectedFoundation === 'calculus' && (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-200">
          <h4 className="text-xl font-bold text-purple-900 mb-4">Calculus in Supervised Learning</h4>
          <p className="text-gray-700 mb-4">
            Used in training: adjusting weights to minimize loss (error).
          </p>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Gradient Descent Algorithm</h5>
              <div className="bg-purple-50 rounded p-3 mb-2">
                <div className="font-mono text-sm mb-2">
                  <strong>θ := θ - α × (∂J(θ) / ∂θ)</strong>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>• α = learning rate (step size)</div>
                  <div>• ∂J(θ)/∂θ = gradient (derivative of loss w.r.t. parameters)</div>
                  <div>• Update parameters in direction opposite to gradient</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Loss Function for Linear Regression</h5>
              <div className="bg-purple-50 rounded p-3 mb-2">
                <div className="font-mono text-sm mb-2 text-center">
                  <strong>J(θ) = (1/2m) × Σ(hθ(xᵢ) - yᵢ)²</strong>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>• m = number of training examples</div>
                  <div>• hθ(xᵢ) = predicted value for sample i</div>
                  <div>• yᵢ = actual value for sample i</div>
                  <div>• This is Mean Squared Error (MSE)</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Gradient Calculation</h5>
              <div className="bg-purple-50 rounded p-3 mb-2">
                <div className="font-mono text-xs space-y-1">
                  <div>∂J(θ)/∂θⱼ = (1/m) × Σ(hθ(xᵢ) - yᵢ) × xᵢⱼ</div>
                  <div className="text-gray-600 mt-2">
                    For all parameters j, compute partial derivative
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-purple-300">
              <h5 className="font-bold text-purple-900 mb-2">Training Process</h5>
              <ol className="list-decimal list-inside space-y-2 text-sm text-gray-700">
                <li>Initialize parameters θ randomly</li>
                <li>Compute predictions: ŷ = Xθ</li>
                <li>Calculate loss: J(θ) = (1/2m) × Σ(ŷ - y)²</li>
                <li>Compute gradient: ∂J(θ)/∂θ</li>
                <li>Update parameters: θ := θ - α × ∂J(θ)/∂θ</li>
                <li>Repeat until convergence (loss stops decreasing)</li>
              </ol>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Calculus enables us to find the optimal parameters by computing derivatives and moving in the direction 
                that minimizes the loss function. This is the mathematical foundation of all optimization algorithms in ML.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Probability */}
      {selectedFoundation === 'probability' && (
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-6 border-2 border-blue-200">
          <h4 className="text-xl font-bold text-blue-900 mb-4">Probability & Statistics in Supervised Learning</h4>
          <p className="text-gray-700 mb-4">
            Helps model uncertainty and interpret predictions.
          </p>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Logistic Regression</h5>
              <p className="text-sm text-gray-700 mb-2">
                Uses the sigmoid function to model probabilities:
              </p>
              <div className="bg-blue-50 rounded p-3 mb-2">
                <div className="font-mono text-sm mb-2 text-center">
                  <strong>P(y = 1 | x) = 1 / (1 + e^(-θᵀx))</strong>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>• Outputs probability between 0 and 1</div>
                  <div>• Sigmoid function: σ(z) = 1 / (1 + e^(-z))</div>
                  <div>• Used for binary classification</div>
                </div>
              </div>
              <div className="bg-gray-100 rounded p-2 font-mono text-xs">
                Decision: Predict y = 1 if P(y = 1 | x) ≥ 0.5, else y = 0
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Naive Bayes Classifier</h5>
              <p className="text-sm text-gray-700 mb-2">
                Uses Bayes' theorem for classification:
              </p>
              <div className="bg-blue-50 rounded p-3 mb-2">
                <div className="font-mono text-sm mb-2">
                  <strong>P(y | x₁, x₂, ..., xₙ) = P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y) / P(x)</strong>
                </div>
                <div className="text-xs text-gray-600 space-y-1 mt-2">
                  <div>• "Naive" assumption: features are conditionally independent</div>
                  <div>• P(y) = prior probability of class y</div>
                  <div>• P(xᵢ|y) = likelihood of feature xᵢ given class y</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Probability Distributions</h5>
              <div className="space-y-2 text-sm text-gray-700">
                <div>
                  <strong>Normal Distribution:</strong> Used to model continuous features in Naive Bayes
                </div>
                <div>
                  <strong>Bernoulli Distribution:</strong> Used for binary features
                </div>
                <div>
                  <strong>Multinomial Distribution:</strong> Used for discrete features with multiple values
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-blue-300">
              <h5 className="font-bold text-blue-900 mb-2">Uncertainty Quantification</h5>
              <p className="text-sm text-gray-700">
                Probability allows us to:
              </p>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 mt-2">
                <li>Quantify prediction confidence</li>
                <li>Make probabilistic predictions (not just binary decisions)</li>
                <li>Handle uncertainty in real-world data</li>
                <li>Interpret model outputs as probabilities</li>
              </ul>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Probability provides a principled way to model uncertainty and make predictions. 
                Instead of just saying "this is class A", we can say "this is class A with 85% confidence", 
                which is much more informative for decision-making.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Optimization */}
      {selectedFoundation === 'optimization' && (
        <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg p-6 border-2 border-orange-200">
          <h4 className="text-xl font-bold text-orange-900 mb-4">Optimization in Supervised Learning</h4>
          <p className="text-gray-700 mb-4">
            Finding the best weights θ using techniques like gradient descent or stochastic gradient descent (SGD).
          </p>

          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Gradient Descent</h5>
              <p className="text-sm text-gray-700 mb-2">
                Iterative optimization algorithm that finds the minimum of a function:
              </p>
              <div className="bg-orange-50 rounded p-3 mb-2">
                <div className="font-mono text-sm mb-2">
                  <strong>θ := θ - α × (∂J(θ) / ∂θ)</strong>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>• α = learning rate (step size)</div>
                  <div>• Moves in direction of steepest descent</div>
                  <div>• Repeats until convergence</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Stochastic Gradient Descent (SGD)</h5>
              <p className="text-sm text-gray-700 mb-2">
                Variant that uses a random subset (mini-batch) of data for each update:
              </p>
              <div className="bg-orange-50 rounded p-3 mb-2">
                <div className="font-mono text-xs space-y-1">
                  <div>For each mini-batch:</div>
                  <div>  θ := θ - α × (∂J(θ) / ∂θ) for mini-batch</div>
                </div>
                <div className="text-xs text-gray-600 space-y-1 mt-2">
                  <div>• Faster updates (uses subset of data)</div>
                  <div>• More noise but can escape local minima</div>
                  <div>• Common in deep learning</div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Convex Optimization</h5>
              <p className="text-sm text-gray-700 mb-2">
                For convex loss functions, optimization guarantees finding the global minimum:
              </p>
              <div className="bg-orange-50 rounded p-3 mb-2">
                <div className="text-xs text-gray-700 space-y-2">
                  <div>
                    <strong>Convex Function:</strong> Any local minimum is also a global minimum
                  </div>
                  <div>
                    <strong>Linear Regression:</strong> MSE loss is convex → guaranteed global minimum
                  </div>
                  <div>
                    <strong>Logistic Regression:</strong> Cross-entropy loss is convex → guaranteed global minimum
                  </div>
                  <div>
                    <strong>Neural Networks:</strong> Non-convex → may find local minima (but often work well in practice)
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Optimization Algorithms</h5>
              <div className="space-y-2 text-sm text-gray-700">
                <div>
                  <strong>Batch Gradient Descent:</strong> Uses all training data for each update (slow but stable)
                </div>
                <div>
                  <strong>Stochastic Gradient Descent (SGD):</strong> Uses one sample at a time (fast but noisy)
                </div>
                <div>
                  <strong>Mini-batch Gradient Descent:</strong> Uses small batches (balance between speed and stability)
                </div>
                <div>
                  <strong>Adam, RMSprop:</strong> Adaptive learning rate methods (faster convergence)
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border-2 border-orange-300">
              <h5 className="font-bold text-orange-900 mb-2">Convergence Criteria</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li>Loss stops decreasing (change &lt; threshold)</li>
                <li>Gradient magnitude becomes very small</li>
                <li>Maximum number of iterations reached</li>
                <li>Validation loss starts increasing (early stopping)</li>
              </ul>
            </div>

            <div className="bg-yellow-50 rounded-lg p-4 border-2 border-yellow-300">
              <h5 className="font-bold text-yellow-900 mb-2">💡 Key Insight</h5>
              <p className="text-sm text-yellow-800">
                Optimization is the process of finding the best model parameters. Convex optimization guarantees 
                finding the global optimum for linear/logistic regression, while non-convex optimization (neural networks) 
                may find local optima but often performs well in practice. The choice of optimization algorithm 
                (SGD, Adam, etc.) significantly affects training speed and final model quality.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

