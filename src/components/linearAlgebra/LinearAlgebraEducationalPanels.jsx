import React from 'react';
import MLUseCases from './MLUseCases';

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
          
          {/* Real-World ML Use Cases */}
          <MLUseCases operationType="determinant" />
        </div>
      )}
    </div>
  );
}

