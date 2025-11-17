import React from 'react';
import MLUseCases from './MLUseCases';
import StepByStepVisualGuide from '../shared/StepByStepVisualGuide';

export default function LinearAlgebraEducationalPanels({ selectedTopic, matrix, matrixA, matrixB }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'eigenvalues' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-indigo-900 mb-3">What are Eigenvalues and Eigenvectors?</h3>
            <p className="text-gray-700 mb-4">
              For a square matrix A, an eigenvector v is a non-zero vector that, when multiplied by A, 
              results in a scalar multiple of itself. This scalar is called the eigenvalue λ.
            </p>
            <div className="bg-indigo-50 rounded-lg p-4 mb-4">
              <div className="font-mono text-lg mb-2">A × v = λ × v</div>
            </div>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Eigenvalues:</strong> Scalars that represent how much the eigenvector is stretched</li>
              <li><strong>Eigenvectors:</strong> Vectors that don't change direction when transformed</li>
              <li><strong>Characteristic Equation:</strong> det(A - λI) = 0</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Them?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                They tell us about the <strong className="text-indigo-700">fundamental behavior</strong> of transformations or systems represented by a matrix.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📏</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Eigenvalues describe scaling</h4>
                    <p className="text-gray-700">
                      Eigenvalues describe <strong>how much each "principal direction" is scaled</strong>. 
                      They quantify the magnitude of transformation along each eigenvector direction.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> If λ = 2, vectors in that direction are stretched by 2×. 
                      If λ = 0.5, they're compressed to half size.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🧭</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Eigenvectors describe directions</h4>
                    <p className="text-gray-700">
                      Eigenvectors describe the <strong>directions in which the system behaves simply</strong> 
                      (without rotation or mixing of components). These are the "natural axes" of the transformation.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> In PCA, eigenvectors point along directions of maximum variance. 
                      In face recognition, eigenfaces represent principal facial feature variations.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg p-4 border-2 border-indigo-300">
                <h4 className="font-bold text-indigo-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Together, eigenvalues and eigenvectors decompose complex matrix transformations into simple, 
                  understandable components. Instead of thinking about how a matrix transforms all vectors, 
                  we can understand it by knowing just a few special directions (eigenvectors) and how much 
                  they're scaled (eigenvalues).
                </p>
              </div>
            </div>
          </div>

          {/* Step-by-Step Guide */}
          <StepByStepVisualGuide
            title="Understanding Eigenvalues and Eigenvectors"
            color="indigo"
            steps={[
              {
                title: "What is an Eigenvector?",
                description: "An eigenvector is a special vector that, when multiplied by a matrix, only gets scaled (stretched or shrunk) but doesn't change direction. Think of it as a direction that the matrix transformation preserves.",
                formula: "A × v = λ × v",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: If v is an eigenvector and λ = 2, then:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      A × v = 2 × v
                    </div>
                    <p className="mt-2">The vector v is stretched by a factor of 2, but points in the same direction.</p>
                  </div>
                )
              },
              {
                title: "Finding Eigenvalues",
                description: "To find eigenvalues, we solve the characteristic equation: det(A - λI) = 0, where I is the identity matrix. The solutions λ are the eigenvalues.",
                formula: "det(A - λI) = 0",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">For a 2×2 matrix A:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      A - λI = [a-λ  b  ]<br />
                               [c    d-λ]
                    </div>
                    <p>Set det(A - λI) = 0 and solve for λ</p>
                  </div>
                )
              },
              {
                title: "Finding Eigenvectors",
                description: "For each eigenvalue λ, solve the system (A - λI)v = 0 to find the corresponding eigenvector v. There may be multiple eigenvectors for the same eigenvalue.",
                formula: "(A - λI) × v = 0",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Steps:</p>
                    <ol className="list-decimal list-inside space-y-1 text-xs">
                      <li>Substitute eigenvalue λ into (A - λI)</li>
                      <li>Solve the homogeneous system</li>
                      <li>The solution space gives the eigenvectors</li>
                    </ol>
                  </div>
                )
              },
              {
                title: "Geometric Interpretation",
                description: "Eigenvectors show the directions along which the transformation acts like a simple scaling. The eigenvalues tell you how much scaling occurs in each direction.",
                formula: "Transformation preserves eigenvector direction",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Visual example:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>If λ = 2: vector is stretched 2× in eigenvector direction</li>
                      <li>If λ = 0.5: vector is compressed to half size</li>
                      <li>If λ = -1: vector is flipped but same magnitude</li>
                    </ul>
                  </div>
                )
              },
              {
                title: "Applications in ML",
                description: "Eigenvalues and eigenvectors are fundamental in Principal Component Analysis (PCA), where they identify the directions of maximum variance in data. The largest eigenvalues correspond to the most important features.",
                formula: "PCA: Find eigenvectors of covariance matrix",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">In PCA:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Compute covariance matrix of data</li>
                      <li>Find eigenvalues and eigenvectors</li>
                      <li>Sort by eigenvalues (largest first)</li>
                      <li>Top k eigenvectors = principal components</li>
                    </ul>
                  </div>
                )
              }
            ]}
          />
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="eigenvalues" />
        </div>
      )}

      {selectedTopic === 'data-representation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-indigo-900 mb-3">Data Representation in ML</h3>
            <p className="text-gray-700 mb-4">
              Machine learning algorithms work with data represented as matrices. Understanding this 
              representation is crucial for implementing and understanding ML models.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Feature Matrix:</strong> Each row is a sample, each column is a feature</li>
              <li><strong>Label Vector:</strong> Target values corresponding to each sample</li>
              <li><strong>Batch Processing:</strong> Processing multiple samples simultaneously using matrix operations</li>
            </ul>
          </div>

          {/* Step-by-Step Guide */}
          <StepByStepVisualGuide
            title="Understanding Data Representation in ML"
            color="indigo"
            steps={[
              {
                title: "Data as a Matrix",
                description: "In ML, data is organized as a matrix where each row represents one data sample (instance) and each column represents one feature (attribute). This structure allows efficient batch processing.",
                formula: "X = [samples × features]",
                visual: (
                  <div className="text-sm text-gray-700">
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      X = [x₁₁  x₁₂  ...  x₁ₙ]  ← Sample 1<br />
                          [x₂₁  x₂₂  ...  x₂ₙ]  ← Sample 2<br />
                          [...  ...  ...  ... ]<br />
                          [xₘ₁  xₘ₂  ...  xₘₙ]  ← Sample m
                    </div>
                    <p className="text-xs">m samples, n features</p>
                  </div>
                )
              },
              {
                title: "Feature Vectors",
                description: "Each row is a feature vector representing one data point. For example, in image classification, each row might be a flattened image (e.g., 28×28 = 784 pixels).",
                formula: "xᵢ = [feature₁, feature₂, ..., featureₙ]",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: Customer data</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      [Age, Income, Purchases, DaysSinceLastPurchase]
                    </div>
                    <p className="mt-2 text-xs">Each sample is a 4-dimensional feature vector</p>
                  </div>
                )
              },
              {
                title: "Label Vector",
                description: "Target values (labels) are stored as a separate vector, with one label per sample. For classification, labels are class indices. For regression, labels are continuous values.",
                formula: "y = [y₁, y₂, ..., yₘ]",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example labels:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      y = [0, 1, 0, 1, ...]  ← Binary classification<br />
                      y = [25.3, 30.1, ...]  ← Regression
                    </div>
                  </div>
                )
              },
              {
                title: "Batch Processing",
                description: "Processing multiple samples at once using matrix multiplication is much faster than processing them one by one. Modern GPUs are optimized for matrix operations.",
                formula: "Batch × Weights = Output",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: Process 32 images at once</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      Input: (32, 784) × Weights: (784, 128) = Output: (32, 128)
                    </div>
                    <p className="mt-2 text-xs">All 32 samples processed simultaneously!</p>
                  </div>
                )
              },
              {
                title: "Normalization and Preprocessing",
                description: "Before feeding data to models, features are often normalized (scaled to similar ranges) to ensure all features contribute equally to the learning process.",
                formula: "x_normalized = (x - mean) / std",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Common preprocessing steps:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Standardization: (x - μ) / σ</li>
                      <li>Min-Max scaling: (x - min) / (max - min)</li>
                      <li>Feature normalization: x / ||x||</li>
                    </ul>
                  </div>
                )
              }
            ]}
          />
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="data-representation" />
        </div>
      )}

      {selectedTopic === 'weight-representation' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-indigo-900 mb-3">Weight Representation in Neural Networks</h3>
            <p className="text-gray-700 mb-4">
              Neural networks store connections between neurons as weight matrices. These matrices 
              are learned during training and determine how information flows through the network.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Weight Matrix Dimensions:</strong> Determined by the number of neurons in connected layers</li>
              <li><strong>Bias Vectors:</strong> Additional parameters added to each layer</li>
              <li><strong>Parameter Count:</strong> Total parameters = (input_size × output_size) + output_size</li>
            </ul>
          </div>

          {/* Step-by-Step Guide */}
          <StepByStepVisualGuide
            title="Understanding Weight Matrices in Neural Networks"
            color="purple"
            steps={[
              {
                title: "What are Weight Matrices?",
                description: "A weight matrix W connects two layers in a neural network. Each element W[i][j] represents the strength of the connection from neuron j in the input layer to neuron i in the output layer.",
                formula: "W: [output_neurons × input_neurons]",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: Connecting 4 inputs to 3 outputs</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      W = [w₁₁  w₁₂  w₁₃  w₁₄]  ← Output neuron 1<br />
                          [w₂₁  w₂₂  w₂₃  w₂₄]  ← Output neuron 2<br />
                          [w₃₁  w₃₂  w₃₃  w₃₄]  ← Output neuron 3
                    </div>
                    <p className="mt-2 text-xs">Shape: (3, 4) = (output_size, input_size)</p>
                  </div>
                )
              },
              {
                title: "Forward Propagation",
                description: "During forward propagation, input is multiplied by the weight matrix, then a bias is added, and finally an activation function is applied.",
                formula: "output = activation(W × input + bias)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Step-by-step:</p>
                    <ol className="list-decimal list-inside space-y-1 text-xs">
                      <li>Multiply: z = W × x</li>
                      <li>Add bias: z = z + b</li>
                      <li>Apply activation: a = σ(z)</li>
                    </ol>
                    <div className="bg-white p-2 rounded font-mono text-xs mt-2">
                      Example: (3,4) × (4,1) + (3,1) = (3,1)
                    </div>
                  </div>
                )
              },
              {
                title: "Weight Initialization",
                description: "Weights are initialized randomly before training. Common methods include Xavier/Glorot initialization (for tanh/sigmoid) and He initialization (for ReLU).",
                formula: "W ~ N(0, σ²) where σ² = 2/(fan_in + fan_out)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Initialization strategies:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Xavier: σ² = 1/(fan_in + fan_out)</li>
                      <li>He: σ² = 2/fan_in (for ReLU)</li>
                      <li>Small random values prevent symmetry breaking</li>
                    </ul>
                  </div>
                )
              },
              {
                title: "Learning Weights",
                description: "During training, weights are updated using gradient descent. The gradient of the loss with respect to weights tells us how to adjust weights to reduce error.",
                formula: "W_new = W_old - α × ∇W_loss",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Training process:</p>
                    <ol className="list-decimal list-inside space-y-1 text-xs">
                      <li>Forward pass: compute predictions</li>
                      <li>Compute loss</li>
                      <li>Backward pass: compute gradients</li>
                      <li>Update weights: W -= learning_rate × gradient</li>
                    </ol>
                  </div>
                )
              },
              {
                title: "Parameter Counting",
                description: "The total number of trainable parameters in a layer equals the number of weights plus biases. This determines model complexity and memory requirements.",
                formula: "Parameters = (input_size × output_size) + output_size",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: Layer with 784 inputs, 128 outputs</p>
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      Weights: 784 × 128 = 100,352<br />
                      Biases: 128<br />
                      Total: 100,480 parameters
                    </div>
                    <p className="text-xs">Large models can have millions or billions of parameters!</p>
                  </div>
                )
              }
            ]}
          />
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="weight-representation" />
        </div>
      )}

      {selectedTopic === 'matrix-operations' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-indigo-900 mb-3">Matrix Operations in ML</h3>
            <p className="text-gray-700 mb-4">
              Matrix operations are fundamental in machine learning. This section covers both 
              <strong> matrix multiplication</strong> (for neural networks) and <strong>matrix transformations</strong> 
              (for geometric operations in computer vision and graphics).
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Matrix Multiplication:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Dimension Requirement:</strong> A (m×n) × B (n×p) = C (m×p)</li>
              <li><strong>Columns of A</strong> must equal <strong>rows of B</strong></li>
              <li><strong>Result dimensions:</strong> Rows from A, Columns from B</li>
              <li><strong>Step-by-Step:</strong> Result[i][j] = Σ(A[i][k] × B[k][j]) for all k</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Matrix Transformations:</h4>
            <p className="text-gray-700 mb-2">
              Matrices can represent geometric transformations like translation, rotation, scaling, and reflection. 
              These transformations are fundamental in computer graphics, computer vision, and data augmentation for ML.
            </p>
            
            <h5 className="font-semibold text-gray-700 mb-2">2D Transformations:</h5>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4 ml-4">
              <li><strong>Translation:</strong> Moves points by (tx, ty). Uses homogeneous coordinates [1, 0, tx; 0, 1, ty; 0, 0, 1]</li>
              <li><strong>Rotation:</strong> Rotates points by angle θ. Matrix: [cos(θ), -sin(θ); sin(θ), cos(θ)]</li>
              <li><strong>Scaling:</strong> Scales by factors sx, sy. Matrix: [sx, 0; 0, sy]</li>
              <li><strong>Reflection:</strong> Reflects across axis. X-axis: [1, 0; 0, -1], Y-axis: [-1, 0; 0, 1]</li>
            </ul>

            <h5 className="font-semibold text-gray-700 mb-2">3D Transformations:</h5>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4 ml-4">
              <li><strong>Rotation:</strong> Rotate around X, Y, or Z axis using rotation matrices</li>
              <li><strong>Scaling:</strong> Scale in 3D space: [sx, 0, 0; 0, sy, 0; 0, 0, sz]</li>
              <li><strong>Translation:</strong> Move in 3D: [1, 0, 0, tx; 0, 1, 0, ty; 0, 0, 1, tz; 0, 0, 0, 1]</li>
            </ul>

            <div className="bg-indigo-50 rounded-lg p-4 mb-4 border-2 border-indigo-200">
              <div className="font-mono text-sm mb-2">
                <div><strong>Homogeneous Coordinates:</strong> Add 1 as third coordinate for 2D, fourth for 3D</div>
                <div className="text-xs text-gray-600 mt-2">
                  Allows translation to be represented as matrix multiplication
                </div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Combining Transformations:</h4>
            <p className="text-gray-700 mb-2">
              Multiple transformations can be combined by multiplying their matrices:
            </p>
            <div className="bg-indigo-50 rounded-lg p-4 mb-4">
              <div className="font-mono text-sm">
                T_combined = T₁ × T₂ × T₃
              </div>
              <div className="text-xs text-gray-600 mt-2">
                Order matters! Matrix multiplication is not commutative.
              </div>
            </div>
          </div>

          {/* Step-by-Step Guide for Matrix Multiplication */}
          <StepByStepVisualGuide
            title="Understanding Matrix Multiplication"
            color="blue"
            steps={[
              {
                title: "Dimension Requirements",
                description: "For matrix multiplication A × B to be valid, the number of columns in A must equal the number of rows in B. The result will have the same number of rows as A and the same number of columns as B.",
                formula: "A (m×n) × B (n×p) = C (m×p)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      (3×4) × (4×2) = (3×2) ✓ Valid<br />
                      (3×4) × (3×2) = ✗ Invalid (4 ≠ 3)
                    </div>
                  </div>
                )
              },
              {
                title: "Element-by-Element Calculation",
                description: "Each element C[i][j] in the result is computed by taking the dot product of row i from A and column j from B. Multiply corresponding elements and sum them up.",
                formula: "C[i][j] = Σ(k=0 to n-1) A[i][k] × B[k][j]",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">For C[0][0]:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      = A[0][0]×B[0][0] + A[0][1]×B[1][0] + A[0][2]×B[2][0] + ...
                    </div>
                    <p className="mt-2 text-xs">Take row 0 from A, column 0 from B, multiply and sum</p>
                  </div>
                )
              },
              {
                title: "Visual Pattern",
                description: "Think of it as: for each cell in the result, take the corresponding row from the first matrix and column from the second matrix, then multiply and add.",
                formula: "Row × Column = Cell",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Pattern:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      Result[0][0] = Row 0 of A × Column 0 of B<br />
                      Result[0][1] = Row 0 of A × Column 1 of B<br />
                      Result[1][0] = Row 1 of A × Column 0 of B<br />
                      ...
                    </div>
                  </div>
                )
              },
              {
                title: "Why Order Matters",
                description: "Matrix multiplication is NOT commutative. A × B ≠ B × A in general. The order determines which transformation is applied first.",
                formula: "A × B ≠ B × A (in general)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      Rotate then Translate ≠ Translate then Rotate
                    </div>
                    <p className="text-xs">In neural networks: input × weights gives forward pass, but weights × input would be wrong!</p>
                  </div>
                )
              },
              {
                title: "Efficiency in ML",
                description: "Matrix multiplication is highly optimized on GPUs. Processing batches of data using matrix multiplication is much faster than processing samples individually.",
                formula: "Batch processing: (batch_size, features) × (features, outputs)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example: Process 32 images at once</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      (32, 784) × (784, 128) = (32, 128)
                    </div>
                    <p className="mt-2 text-xs">All 32 samples processed in parallel, much faster than 32 individual operations!</p>
                  </div>
                )
              }
            ]}
          />
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="matrix-multiplication" />
          <div className="mt-6">
            <MLUseCases operationType="matrix-transformations" />
          </div>
        </div>
      )}

      {selectedTopic === 'determinant' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-indigo-900 mb-3">What is a Determinant?</h3>
            <p className="text-gray-700 mb-4">
              The determinant is a scalar value that can be computed from a square matrix. It provides 
              crucial information about the matrix, including whether it's invertible and how it scales volumes.
            </p>
            
            <div className="bg-indigo-50 rounded-lg p-4 mb-4">
              <div className="font-mono text-lg mb-2">
                For a 2×2 matrix: det(A) = a₁₁ × a₂₂ - a₁₂ × a₂₁
              </div>
              <div className="font-mono text-sm mt-2">
                For a 3×3 matrix: det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">Key Properties:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Invertibility:</strong> A matrix is invertible if and only if det(A) ≠ 0</li>
              <li><strong>Volume Scaling:</strong> |det(A)| = factor by which A scales volumes</li>
              <li><strong>Product Rule:</strong> det(AB) = det(A) × det(B)</li>
              <li><strong>Transpose:</strong> det(A) = det(A^T)</li>
              <li><strong>Singular Matrix:</strong> det(A) = 0 means columns are linearly dependent</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Determinants?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Determinants provide <strong className="text-indigo-700">critical information</strong> about matrices that is essential for understanding 
                matrix behavior, solving systems of equations, and ensuring numerical stability in ML algorithms.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔓</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Invertibility Check</h4>
                    <p className="text-gray-700">
                      Determinants tell us <strong>whether a matrix can be inverted</strong>. 
                      If det(A) = 0, the matrix is singular (non-invertible), which means we cannot solve 
                      systems of equations uniquely or compute matrix inverses.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> In ML, we need invertible matrices for solving linear systems, 
                      computing covariance matrix inverses, and ensuring numerical stability in optimization.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📐</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Geometric Interpretation</h4>
                    <p className="text-gray-700">
                      The absolute value of the determinant represents the <strong>scaling factor for areas/volumes</strong>. 
                      It tells us how much a transformation stretches or compresses space.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> In 2D, |det(A)| = area of parallelogram formed by column vectors. 
                      In 3D, |det(A)| = volume of parallelepiped. If det = 0, vectors are collinear/coplanar.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔗</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Linear Independence Detection</h4>
                    <p className="text-gray-700">
                      Determinants reveal <strong>whether columns (or rows) are linearly independent</strong>. 
                      If det(A) = 0, the columns are linearly dependent, meaning some features are redundant.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> In feature engineering, det(covariance_matrix) = 0 indicates 
                      multicollinearity - redundant features that should be removed to improve model performance.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔢</div>
                  <div>
                    <h4 className="font-bold text-indigo-900 mb-2">Connection to Eigenvalues</h4>
                    <p className="text-gray-700">
                      The determinant equals the <strong>product of all eigenvalues</strong>. 
                      This provides a quick way to check if a matrix has a zero eigenvalue, which would make det = 0.
                    </p>
                    <div className="mt-2 bg-indigo-50 rounded p-2 text-sm text-indigo-800">
                      <strong>Example:</strong> det(A) = λ₁ × λ₂ × ... × λₙ. If any eigenvalue is 0, 
                      then det(A) = 0, indicating the matrix is singular.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg p-4 border-2 border-indigo-300">
                <h4 className="font-bold text-indigo-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Determinants serve as a <strong>single number that captures essential matrix properties</strong>. 
                  They act as a "health check" for matrices - telling us if a matrix is invertible, if its columns 
                  are independent, and how it transforms geometric space. In ML, checking determinants helps ensure 
                  numerical stability, detect feature redundancy, and validate that our computations are well-defined.
                </p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-800 mb-2">Geometric Interpretation:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>2D:</strong> Absolute value = area of parallelogram formed by column vectors</li>
              <li><strong>3D:</strong> Absolute value = volume of parallelepiped formed by column vectors</li>
              <li><strong>Sign:</strong> Positive = preserves orientation, Negative = reverses orientation</li>
              <li><strong>Zero:</strong> Vectors are collinear (2D) or coplanar (3D)</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Definition:</h4>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <p className="text-gray-700 text-sm mb-2">
                The determinant can be computed using:
              </p>
              <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
                <li><strong>Laplace Expansion:</strong> Expand along any row or column</li>
                <li><strong>Sarrus' Rule:</strong> Special method for 3×3 matrices</li>
                <li><strong>LU Decomposition:</strong> det(A) = det(L) × det(U) = product of diagonal elements</li>
                <li><strong>Eigenvalues:</strong> det(A) = product of all eigenvalues</li>
              </ul>
            </div>
          </div>

          <div>
            {/* Connection to Eigenvalues */}
            <div className="bg-indigo-50 rounded-lg p-4 border-2 border-indigo-200 mb-4">
              <h4 className="font-semibold text-indigo-900 mb-3">Connection to Eigenvalues:</h4>
              <div className="space-y-3">
                <div className="bg-white rounded p-3 border border-indigo-300">
                  <div className="font-mono text-sm mb-2 font-semibold text-indigo-900">
                    det(A) = λ₁ × λ₂ × ... × λₙ
                  </div>
                  <p className="text-gray-700 text-sm mb-2">
                    The determinant of a matrix equals the product of all its eigenvalues.
                  </p>
                  <div className="text-xs text-gray-600 space-y-1">
                    <div>• This is a fundamental relationship in linear algebra</div>
                    <div>• If det(A) = 0, then at least one eigenvalue is 0</div>
                    <div>• Characteristic polynomial: det(A - λI) = 0</div>
                    <div>• Eigenvalues are roots of the characteristic polynomial</div>
                  </div>
                </div>
                <div className="bg-indigo-100 rounded p-3">
                  <div className="text-sm text-indigo-900 font-semibold mb-1">Example:</div>
                  <div className="text-xs text-indigo-800 font-mono">
                    If A has eigenvalues λ₁ = 2, λ₂ = 3, λ₃ = 4<br />
                    Then det(A) = 2 × 3 × 4 = 24
                  </div>
                </div>
              </div>
            </div>

            {/* Connection to Matrix Invertibility */}
            <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200 mb-4">
              <h4 className="font-semibold text-green-900 mb-3">Connection to Matrix Invertibility:</h4>
              <div className="space-y-3">
                <div className="bg-white rounded p-3 border border-green-300">
                  <div className="font-mono text-sm mb-2 font-semibold text-green-900">
                    A is invertible ⟺ det(A) ≠ 0
                  </div>
                  <p className="text-gray-700 text-sm mb-2">
                    A square matrix is invertible if and only if its determinant is non-zero.
                  </p>
                  <div className="text-xs text-gray-600 space-y-1">
                    <div>• <strong>If det(A) ≠ 0:</strong> Matrix is invertible, columns are linearly independent</div>
                    <div>• <strong>If det(A) = 0:</strong> Matrix is singular, columns are linearly dependent</div>
                    <div>• <strong>Inverse formula:</strong> A⁻¹ = (1/det(A)) × adj(A)</div>
                    <div>• <strong>Connection to eigenvalues:</strong> det(A) = 0 ⟺ at least one eigenvalue is 0</div>
                  </div>
                </div>
                <div className="bg-green-100 rounded p-3">
                  <div className="text-sm text-green-900 font-semibold mb-1">Why This Matters in ML:</div>
                  <div className="text-xs text-green-800 space-y-1">
                    <div>• <strong>Solving Linear Systems:</strong> Ax = b requires A to be invertible</div>
                    <div>• <strong>Numerical Stability:</strong> Near-zero determinants cause numerical errors</div>
                    <div>• <strong>Regularization:</strong> L2 regularization ensures matrices stay invertible</div>
                    <div>• <strong>Covariance Matrices:</strong> Must be invertible for multivariate analysis</div>
                    <div>• <strong>Feature Selection:</strong> Remove features that make covariance matrix singular</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Unified Relationship */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4 border-2 border-purple-200">
              <h4 className="font-semibold text-purple-900 mb-3">Unified Relationship:</h4>
              <div className="bg-white rounded p-3 border border-purple-300">
                <div className="text-sm text-purple-900 font-semibold mb-2">
                  The Three Pillars of Matrix Analysis:
                </div>
                <div className="space-y-2 text-xs text-gray-700">
                  <div className="flex items-start gap-2">
                    <span className="font-bold text-purple-600">1.</span>
                    <div>
                      <strong>Determinant:</strong> det(A) = product of eigenvalues
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="font-bold text-purple-600">2.</span>
                    <div>
                      <strong>Invertibility:</strong> A⁻¹ exists ⟺ det(A) ≠ 0
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="font-bold text-purple-600">3.</span>
                    <div>
                      <strong>Eigenvalues:</strong> det(A) = 0 ⟺ at least one eigenvalue is 0
                    </div>
                  </div>
                </div>
                <div className="mt-3 pt-2 border-t border-purple-200">
                  <div className="text-xs text-purple-800 font-mono">
                    det(A) = 0 ⟺ A is singular ⟺ A has zero eigenvalue ⟺ A is not invertible
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Step-by-Step Guide for Determinant */}
          <StepByStepVisualGuide
            title="Understanding Determinants"
            color="indigo"
            steps={[
              {
                title: "What is a Determinant?",
                description: "The determinant is a scalar value computed from a square matrix. It tells us about the matrix's properties: whether it's invertible, how it scales volumes, and if its columns are linearly independent.",
                formula: "det(A) = scalar value",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Key facts:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Only defined for square matrices</li>
                      <li>det(A) = 0 means matrix is singular (not invertible)</li>
                      <li>det(A) ≠ 0 means matrix is invertible</li>
                    </ul>
                  </div>
                )
              },
              {
                title: "2×2 Matrix Formula",
                description: "For a 2×2 matrix, the determinant is computed as the product of the main diagonal minus the product of the off-diagonal.",
                formula: "det(A) = a₁₁ × a₂₂ - a₁₂ × a₂₁",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Visual pattern:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      [a  b]  det = a×d - b×c<br />
                      [c  d]
                    </div>
                    <p className="mt-2 text-xs">Multiply diagonal ↘, subtract diagonal ↙</p>
                  </div>
                )
              },
              {
                title: "3×3 Matrix Formula",
                description: "For a 3×3 matrix, use Sarrus' rule or cofactor expansion. Sarrus' rule extends the diagonal pattern, while cofactor expansion expands along a row or column.",
                formula: "det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Cofactor expansion along first row:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs">
                      = a₁₁×det(minor₁₁) - a₁₂×det(minor₁₂) + a₁₃×det(minor₁₃)
                    </div>
                    <p className="mt-2 text-xs">Alternating signs: +, -, +</p>
                  </div>
                )
              },
              {
                title: "Geometric Interpretation",
                description: "The absolute value of the determinant equals the area (2D) or volume (3D) of the parallelogram/parallelepiped formed by the column vectors. The sign indicates orientation.",
                formula: "|det(A)| = area/volume, sign = orientation",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">2D Example:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>Column vectors form a parallelogram</li>
                      <li>|det| = area of parallelogram</li>
                      <li>det = 0 means vectors are collinear (no area)</li>
                    </ul>
                  </div>
                )
              },
              {
                title: "Connection to Invertibility",
                description: "A matrix is invertible if and only if its determinant is non-zero. If det(A) = 0, the matrix is singular and cannot be inverted. This happens when columns are linearly dependent.",
                formula: "A is invertible ⟺ det(A) ≠ 0",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Why this matters:</p>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li>det(A) = 0 → columns are linearly dependent</li>
                      <li>det(A) = 0 → matrix is singular</li>
                      <li>det(A) = 0 → cannot solve Ax = b uniquely</li>
                      <li>det(A) ≠ 0 → matrix is invertible</li>
                    </ul>
                  </div>
                )
              },
              {
                title: "Connection to Eigenvalues",
                description: "The determinant equals the product of all eigenvalues. This provides a quick way to check if a matrix has a zero eigenvalue (which would make det = 0).",
                formula: "det(A) = λ₁ × λ₂ × ... × λₙ",
                visual: (
                  <div className="text-sm text-gray-700">
                    <p className="mb-2">Example:</p>
                    <div className="bg-white p-2 rounded font-mono text-xs mb-2">
                      If eigenvalues are 2, 3, 4:<br />
                      det(A) = 2 × 3 × 4 = 24
                    </div>
                    <p className="text-xs">If any eigenvalue is 0, then det(A) = 0</p>
                  </div>
                )
              }
            ]}
          />
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="determinant" />
        </div>
      )}
    </div>
  );
}

