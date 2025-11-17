import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function CalculusEducationalPanels({ selectedTopic, functionType }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'derivatives' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Derivatives in Machine Learning</h3>
            <p className="text-gray-700 mb-4">
              Derivatives measure the rate of change of a function. In ML, they're used to find 
              optimal parameters by minimizing loss functions.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Slope:</strong> The derivative at a point gives the slope of the tangent line</li>
              <li><strong>Optimization:</strong> Finding where derivative equals zero locates minima/maxima</li>
              <li><strong>Gradient Descent:</strong> Uses derivatives to iteratively minimize loss functions</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Derivatives?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Derivatives are the <strong className="text-purple-700">foundation of optimization</strong> in machine learning. 
                They tell us how to adjust model parameters to improve performance and find the best solutions.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📉</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Finding Optimal Parameters</h4>
                    <p className="text-gray-700">
                      Derivatives tell us the <strong>direction and magnitude of change</strong>. When the derivative is zero, 
                      we've found a minimum or maximum - the optimal parameter values that minimize loss or maximize performance.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> If dL/dθ = 0, we've found the optimal θ that minimizes the loss function L.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🧭</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Direction of Improvement</h4>
                    <p className="text-gray-700">
                      Derivatives show us <strong>which direction to move</strong> to improve our model. 
                      The sign tells us whether to increase or decrease parameters, and the magnitude tells us how steep the change is.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> If dL/dw = -5, decreasing w will reduce loss. If dL/dw = +3, increasing w will reduce loss.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚡</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Efficient Learning</h4>
                    <p className="text-gray-700">
                      Without derivatives, we'd have to try random parameter values (inefficient!). 
                      Derivatives give us a <strong>systematic way to improve</strong> - we know exactly how to adjust parameters 
                      to get better results.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> Gradient descent uses derivatives to converge to optimal parameters in hundreds of iterations 
                      instead of millions of random trials.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-4 border-2 border-purple-300">
                <h4 className="font-bold text-purple-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Derivatives transform the problem of "finding the best model" from a search problem into a 
                  <strong> guided optimization problem</strong>. Instead of guessing, derivatives tell us exactly 
                  how to adjust parameters to improve performance. This is why every ML training algorithm relies on derivatives.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="derivatives" />
        </div>
      )}

      {selectedTopic === 'partial-derivatives' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Partial Derivatives</h3>
            <p className="text-gray-700 mb-4">
              Partial derivatives measure how a multivariable function changes with respect to one 
              variable while keeping others constant. Essential for understanding gradients.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">ML Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Multi-parameter Optimization:</strong> Each parameter has its own partial derivative</li>
              <li><strong>Gradient Computation:</strong> Gradients are vectors of partial derivatives</li>
              <li><strong>Feature Importance:</strong> Partial derivatives show feature sensitivity</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Partial Derivatives?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Partial derivatives allow us to optimize <strong className="text-purple-700">multivariable functions</strong> 
                by understanding how each variable independently affects the outcome. In ML, models have many parameters, 
                and we need to know how to adjust each one.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎛️</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Individual Parameter Control</h4>
                    <p className="text-gray-700">
                      Partial derivatives tell us <strong>how each parameter independently affects the loss</strong>. 
                      This allows us to adjust each weight or bias separately, optimizing the entire model systematically.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> ∂L/∂w₁ tells us how changing weight w₁ affects loss, while keeping all other weights fixed.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Building Gradients</h4>
                    <p className="text-gray-700">
                      Gradients are <strong>vectors of partial derivatives</strong>. Each component is a partial derivative 
                      with respect to one parameter. This gives us the complete picture of how to adjust all parameters simultaneously.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> ∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ] - the gradient combines all partial derivatives.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔍</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Feature Sensitivity Analysis</h4>
                    <p className="text-gray-700">
                      Partial derivatives reveal <strong>which features matter most</strong> for predictions. 
                      Large partial derivatives indicate sensitive features that significantly impact the model output.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> If ∂L/∂feature₁ is large, small changes in feature₁ cause big changes in loss - 
                      this feature is important for the model.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-4 border-2 border-purple-300">
                <h4 className="font-bold text-purple-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  In ML, we rarely optimize single variables - we optimize hundreds, thousands, or millions of parameters. 
                  Partial derivatives let us <strong>decompose the complex optimization problem</strong> into simpler 
                  one-dimensional problems. By optimizing each parameter independently (using its partial derivative), 
                  we can efficiently optimize the entire model.
                </p>
              </div>
            </div>
          </div>

          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="partial-derivatives" />
        </div>
      )}

      {selectedTopic === 'gradients' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Gradients and Gradient Descent</h3>
            <p className="text-gray-700 mb-4">
              The gradient is a vector pointing in the direction of steepest ascent. Gradient descent 
              moves in the opposite direction to minimize loss functions.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Gradient Descent Algorithm:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li>Initialize parameters randomly</li>
              <li>Compute gradient of loss function</li>
              <li>Update parameters: θ = θ - α∇L(θ)</li>
              <li>Repeat until convergence</li>
            </ol>
            <p className="text-gray-700 mb-4">
              Where α is the learning rate controlling step size.
            </p>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Gradients?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Gradients provide the <strong className="text-purple-700">optimal direction</strong> for improving 
                our models. They combine information from all parameters to tell us the best way to adjust everything simultaneously.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🧗</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Steepest Descent Direction</h4>
                    <p className="text-gray-700">
                      The gradient points in the direction of <strong>steepest ascent</strong> (fastest increase). 
                      By moving in the opposite direction (-gradient), we take the <strong>steepest descent</strong> - 
                      the fastest way to reduce loss.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> On a 3D loss surface, the gradient tells us which direction slopes down most steeply - 
                      that's the fastest path to the minimum.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Coordinated Parameter Updates</h4>
                    <p className="text-gray-700">
                      Gradients coordinate <strong>all parameter adjustments simultaneously</strong>. Instead of adjusting 
                      parameters one by one (slow!), gradients tell us how to adjust all parameters together for maximum improvement.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> A neural network with 1 million weights - gradients compute all 1 million 
                      updates in one step, making training feasible.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚖️</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Balanced Optimization</h4>
                    <p className="text-gray-700">
                      Gradients ensure we <strong>balance improvements across all parameters</strong>. 
                      They prevent us from over-optimizing one parameter while ignoring others, leading to better overall model performance.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> If adjusting weight w₁ reduces loss by 10 and w₂ reduces loss by 5, 
                      the gradient tells us to adjust w₁ more than w₂, balancing the optimization.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-4 border-2 border-purple-300">
                <h4 className="font-bold text-purple-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Gradients are the <strong>GPS for optimization</strong>. Just like GPS tells you the best route considering 
                  all roads simultaneously, gradients tell you the best way to adjust all parameters simultaneously. 
                  Without gradients, training neural networks would be like navigating without GPS - possible but extremely slow and inefficient.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="gradients" />
        </div>
      )}

      {selectedTopic === 'chain-rule' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Chain Rule</h3>
            <p className="text-gray-700 mb-4">
              The chain rule enables computing derivatives of composite functions. This is the 
              mathematical foundation of backpropagation in neural networks.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Why It Matters:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Neural Networks:</strong> Composed of multiple layers (composite functions)</li>
              <li><strong>Backpropagation:</strong> Uses chain rule to compute gradients layer by layer</li>
              <li><strong>Efficiency:</strong> Allows efficient gradient computation for deep networks</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need the Chain Rule?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                The chain rule enables us to compute derivatives of <strong className="text-purple-700">composite functions</strong> 
                by breaking them down into simpler parts. Neural networks are composite functions, so the chain rule is essential for training them.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔗</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Breaking Down Complexity</h4>
                    <p className="text-gray-700">
                      Neural networks are <strong>chains of functions</strong>: input → layer1 → layer2 → ... → output. 
                      The chain rule lets us compute the derivative of the entire network by multiplying derivatives of each layer.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> If output = f(g(h(x))), then d(output)/dx = (df/dg) × (dg/dh) × (dh/dx) - 
                      the chain rule multiplies derivatives.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">⚡</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Efficient Gradient Computation</h4>
                    <p className="text-gray-700">
                      Without the chain rule, we'd have to compute derivatives of the entire network as one giant function 
                      (impossible for deep networks!). The chain rule lets us <strong>compute gradients layer by layer</strong>, 
                      reusing intermediate results.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> In backpropagation, we compute gradients backward: output → hidden2 → hidden1 → input, 
                      reusing computations at each step.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🏗️</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Enabling Deep Learning</h4>
                    <p className="text-gray-700">
                      The chain rule makes <strong>deep neural networks trainable</strong>. It allows us to propagate 
                      gradients through hundreds of layers efficiently, which is why we can train ResNet-152, GPT-3, and other deep models.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> A 100-layer network requires 100 chain rule multiplications to compute gradients - 
                      still efficient and feasible!
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-4 border-2 border-purple-300">
                <h4 className="font-bold text-purple-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  The chain rule transforms the problem of "computing gradients for a complex network" into 
                  <strong> computing simple derivatives for each layer and multiplying them</strong>. This decomposition 
                  is what makes backpropagation possible. Without the chain rule, we couldn't train neural networks efficiently, 
                  and modern AI wouldn't exist.
                </p>
              </div>
            </div>
          </div>

          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="chain-rule" />
        </div>
      )}

      {selectedTopic === 'backpropagation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Backpropagation</h3>
            <p className="text-gray-700 mb-4">
              Backpropagation is the algorithm used to train neural networks. It efficiently computes 
              gradients for all weights using the chain rule.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">How It Works:</h4>
            <ol className="list-decimal list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Forward Pass:</strong> Compute network output for given input</li>
              <li><strong>Compute Loss:</strong> Compare output to target</li>
              <li><strong>Backward Pass:</strong> Propagate error backward using chain rule</li>
              <li><strong>Update Weights:</strong> Adjust weights using computed gradients</li>
            </ol>
            <p className="text-gray-700 mb-4">
              This process is repeated for many iterations until the network learns.
            </p>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Backpropagation?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Backpropagation is the <strong className="text-purple-700">algorithm that makes neural networks learn</strong>. 
                It efficiently computes gradients for millions of parameters, enabling training of deep learning models.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🚀</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Efficient Gradient Computation</h4>
                    <p className="text-gray-700">
                      Backpropagation computes gradients for <strong>all weights in one backward pass</strong>. 
                      Without it, we'd need to compute gradients separately for each weight (millions of times slower!).
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> A network with 1M weights - backpropagation computes all 1M gradients in one pass, 
                      not 1M separate computations.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔄</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Reusing Computations</h4>
                    <p className="text-gray-700">
                      Backpropagation <strong>reuses intermediate values</strong> from the forward pass. 
                      This makes it much more efficient than computing gradients from scratch for each weight.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> Activations computed during forward pass are reused in backward pass - 
                      no need to recompute them.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🧠</div>
                  <div>
                    <h4 className="font-bold text-purple-900 mb-2">Enabling Deep Learning</h4>
                    <p className="text-gray-700">
                      Backpropagation makes <strong>deep neural networks trainable</strong>. It propagates error signals 
                      backward through many layers, allowing networks to learn complex hierarchical representations.
                    </p>
                    <div className="mt-2 bg-purple-50 rounded p-2 text-sm text-purple-800">
                      <strong>Example:</strong> GPT-3 has 175 billion parameters across 96 layers - backpropagation trains 
                      all of them efficiently, enabling modern AI.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg p-4 border-2 border-purple-300">
                <h4 className="font-bold text-purple-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Backpropagation is the <strong>engine of deep learning</strong>. It transforms the seemingly impossible 
                  task of training networks with millions of parameters into a feasible optimization problem. 
                  Without backpropagation, neural networks would be untrainable, and modern AI (image recognition, 
                  language models, self-driving cars) wouldn't exist.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="calculus" operationType="backpropagation" />
        </div>
      )}
    </div>
  );
}

