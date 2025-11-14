import React, { useMemo, useState, useEffect } from 'react';
import { calculateEigenDecomposition } from '../../utils/linearAlgebra';
import * as math from '../../utils/math';
import InteractiveEigenvalueVisualization from './InteractiveEigenvalueVisualization';
import InteractiveDataRepresentation from './InteractiveDataRepresentation';
import InteractiveWeightVisualization from './InteractiveWeightVisualization';
import InteractiveDeterminantMatrix from './InteractiveDeterminantMatrix';
import InteractiveMatrixTransformationsVisualization from './InteractiveMatrixTransformationsVisualization';

export default function LinearAlgebraVisualization({ 
  selectedTopic, 
  matrix,
  matrixA,
  matrixB,
  matrixSizeA,
  matrixSizeB
}) {
  const [highlightedRow, setHighlightedRow] = useState(null);
  const [highlightedCol, setHighlightedCol] = useState(null);
  const [activeStep, setActiveStep] = useState(null);
  const [matrixOperationType, setMatrixOperationType] = useState('multiplication');

  const eigenDecomp = useMemo(() => {
    if (selectedTopic === 'eigenvalues') {
      try {
        return calculateEigenDecomposition(matrix);
      } catch (e) {
        return null;
      }
    }
    return null;
  }, [selectedTopic, matrix]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'eigenvalues' && eigenDecomp && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Eigenvalues</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              {Array.isArray(eigenDecomp.eigenvalues) ? (
                eigenDecomp.eigenvalues.map((eigenvalue, i) => (
                  <div key={i} className="mb-2">
                    {typeof eigenvalue === 'object' && eigenvalue.imag !== undefined ? (
                      <span className="font-mono text-lg">
                        λ{i + 1} = {eigenvalue.real.toFixed(3)} {eigenvalue.imag >= 0 ? '+' : '-'} {Math.abs(eigenvalue.imag).toFixed(3)}i
                      </span>
                    ) : (
                      <span className="font-mono text-lg">
                        λ{i + 1} = {eigenvalue.toFixed(3)}
                      </span>
                    )}
                  </div>
                ))
              ) : (
                <span className="font-mono text-lg">No eigenvalues calculated</span>
              )}
            </div>
          </div>

          {eigenDecomp.eigenvectors && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Eigenvectors</h3>
              <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                {eigenDecomp.eigenvectors.map((ev, i) => (
                  <div key={i}>
                    <div className="text-sm text-gray-600 mb-1">
                      For λ{i + 1} = {ev.eigenvalue.toFixed(3)}:
                    </div>
                    <div className="font-mono">
                      v{i + 1} = ({ev.eigenvector.x.toFixed(3)}, {ev.eigenvector.y.toFixed(3)})
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Interactive Visualization */}
          <div className="mt-6">
            <InteractiveEigenvalueVisualization 
              matrix={matrix}
              eigenDecomp={eigenDecomp}
            />
          </div>
        </div>
      )}

      {selectedTopic === 'data-representation' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800">Data Matrix Representation</h3>
          
          {/* Interactive Data Representation */}
          <InteractiveDataRepresentation />

          {/* Additional Info */}
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-gray-700 mb-4">
              In ML, data is typically represented as a matrix where:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Rows represent individual samples/data points</li>
              <li>Columns represent features/variables</li>
              <li>Each entry is a feature value for a specific sample</li>
            </ul>
            <div className="mt-4 bg-white p-4 rounded border-2 border-indigo-200">
              <div className="font-mono text-sm">
                <div>X = [</div>
                <div className="ml-4">[x₁₁, x₁₂, ..., x₁ₙ],  ← Sample 1</div>
                <div className="ml-4">[x₂₁, x₂₂, ..., x₂ₙ],  ← Sample 2</div>
                <div className="ml-4">...</div>
                <div className="ml-4">[xₘ₁, xₘ₂, ..., xₘₙ]   ← Sample m</div>
                <div>]</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedTopic === 'weight-representation' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800">Neural Network Weight Representation</h3>
          
          {/* Interactive Weight Visualization */}
          <InteractiveWeightVisualization />

          {/* Additional Info */}
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-gray-700 mb-4">
              In neural networks, weights between layers are represented as matrices:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>Each weight matrix connects two layers</li>
              <li>Rows correspond to neurons in the output layer</li>
              <li>Columns correspond to neurons in the input layer</li>
              <li>Weight W[i][j] connects input neuron j to output neuron i</li>
            </ul>
            <div className="mt-4 bg-white p-4 rounded border-2 border-indigo-200">
              <div className="font-mono text-sm">
                <div>W = [</div>
                <div className="ml-4">[w₁₁, w₁₂, ..., w₁ₙ],  ← Output neuron 1</div>
                <div className="ml-4">[w₂₁, w₂₂, ..., w₂ₙ],  ← Output neuron 2</div>
                <div className="ml-4">...</div>
                <div className="ml-4">[wₘ₁, wₘ₂, ..., wₘₙ]   ← Output neuron m</div>
                <div>]</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedTopic === 'determinant' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Determinant Calculation</h3>
          
          {/* Interactive Matrix Display */}
          <InteractiveDeterminantMatrix matrix={matrix} />

          {/* Determinant Result */}
          {(() => {
            const isSquare = matrix.length > 0 && matrix.length === matrix[0].length;
            if (!isSquare) {
              return (
                <div className="bg-red-50 rounded-lg p-4 border-2 border-red-200">
                  <p className="text-red-900 font-semibold">
                    ⚠️ Determinant is only defined for square matrices!
                  </p>
                  <p className="text-red-700 text-sm mt-2">
                    Current matrix: {matrix.length}×{matrix[0]?.length || 0}
                  </p>
                </div>
              );
            }

            const det = math.determinant(matrix);
            const isSingular = Math.abs(det) < 1e-10;

            return (
              <div className="space-y-4">
                {/* Result Display */}
                <div className={`rounded-lg p-6 border-2 ${
                  isSingular 
                    ? 'bg-yellow-50 border-yellow-300' 
                    : 'bg-green-50 border-green-300'
                }`}>
                  <div className="text-center">
                    <div className="text-sm font-semibold text-gray-700 mb-2">
                      Determinant
                    </div>
                    <div className="text-4xl font-bold font-mono mb-2">
                      det(A) = {det.toFixed(6)}
                    </div>
                    {isSingular && (
                      <div className="text-yellow-800 font-semibold mt-2">
                        ⚠️ This matrix is singular (non-invertible)
                      </div>
                    )}
                    {!isSingular && (
                      <div className="text-green-800 font-semibold mt-2">
                        ✓ This matrix is invertible
                      </div>
                    )}
                  </div>
                </div>

                {/* Step-by-Step Calculation for 2x2 and 3x3 */}
                {matrix.length === 2 && (
                  <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                    <h5 className="font-semibold text-blue-900 mb-3">Step-by-Step Calculation (2×2):</h5>
                    <div className="bg-white rounded p-3 space-y-2">
                      <div className="font-mono text-sm">
                        <div className="mb-2">det(A) = a₁₁ × a₂₂ - a₁₂ × a₂₁</div>
                        <div className="text-blue-700">
                          = {matrix[0][0].toFixed(2)} × {matrix[1][1].toFixed(2)} - {matrix[0][1].toFixed(2)} × {matrix[1][0].toFixed(2)}
                        </div>
                        <div className="text-green-700 font-bold mt-2">
                          = {(matrix[0][0] * matrix[1][1]).toFixed(2)} - {(matrix[0][1] * matrix[1][0]).toFixed(2)}
                        </div>
                        <div className="text-indigo-900 font-bold text-lg mt-2">
                          = {det.toFixed(6)}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {matrix.length === 3 && (
                  <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                    <h5 className="font-semibold text-blue-900 mb-3">Step-by-Step Calculation (3×3):</h5>
                    <div className="bg-white rounded p-3 space-y-2">
                      <div className="font-mono text-sm">
                        <div className="mb-2">Using Sarrus' rule or cofactor expansion:</div>
                        <div className="text-blue-700 space-y-1">
                          <div>det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)</div>
                          <div className="mt-2">
                            = {matrix[0][0].toFixed(2)} × ({matrix[1][1].toFixed(2)} × {matrix[2][2].toFixed(2)} - {matrix[1][2].toFixed(2)} × {matrix[2][1].toFixed(2)})
                            {' - '}
                            {matrix[0][1].toFixed(2)} × ({matrix[1][0].toFixed(2)} × {matrix[2][2].toFixed(2)} - {matrix[1][2].toFixed(2)} × {matrix[2][0].toFixed(2)})
                            {' + '}
                            {matrix[0][2].toFixed(2)} × ({matrix[1][0].toFixed(2)} × {matrix[2][1].toFixed(2)} - {matrix[1][1].toFixed(2)} × {matrix[2][0].toFixed(2)})
                          </div>
                        </div>
                        <div className="text-green-700 font-bold mt-2">
                          = {det.toFixed(6)}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Geometric Interpretation */}
                <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200">
                  <h5 className="font-semibold text-purple-900 mb-3">Geometric Interpretation:</h5>
                  <div className="space-y-2 text-sm text-purple-800">
                    <div>• <strong>2D:</strong> Absolute value = area of parallelogram formed by column vectors</div>
                    <div>• <strong>3D:</strong> Absolute value = volume of parallelepiped formed by column vectors</div>
                    <div>• <strong>Sign:</strong> Positive = orientation preserved, Negative = orientation reversed</div>
                    {det === 0 && (
                      <div className="text-red-700 font-semibold mt-2">
                        • <strong>Zero determinant:</strong> Vectors are linearly dependent (collinear/coplanar)
                      </div>
                    )}
                  </div>
                </div>

                {/* Connection to Eigenvalues */}
                {(() => {
                  try {
                    const eigenDecomp = calculateEigenDecomposition(matrix);
                    if (eigenDecomp && eigenDecomp.eigenvalues) {
                      const eigenvalues = eigenDecomp.eigenvalues;
                      const productOfEigenvalues = Array.isArray(eigenvalues) 
                        ? eigenvalues.reduce((prod, ev) => {
                            if (typeof ev === 'object' && ev.imag !== undefined) {
                              // Complex eigenvalue: |λ|² = real² + imag²
                              return prod * (ev.real * ev.real + ev.imag * ev.imag);
                            }
                            return prod * ev;
                          }, 1)
                        : eigenvalues;
                      
                      return (
                        <div className="bg-indigo-50 rounded-lg p-4 border-2 border-indigo-200">
                          <h5 className="font-semibold text-indigo-900 mb-3">Connection to Eigenvalues:</h5>
                          <div className="space-y-3">
                            <div className="bg-white rounded p-3 border border-indigo-300">
                              <div className="font-mono text-sm mb-2 font-semibold text-indigo-900">
                                det(A) = λ₁ × λ₂ × ... × λₙ
                              </div>
                              <div className="text-xs text-gray-700 mb-2">
                                The determinant equals the product of all eigenvalues
                              </div>
                              <div className="text-sm text-indigo-800">
                                <div className="mb-1">Eigenvalues:</div>
                                {Array.isArray(eigenvalues) ? (
                                  eigenvalues.map((ev, i) => (
                                    <div key={i} className="font-mono text-xs">
                                      λ{i + 1} = {typeof ev === 'object' && ev.imag !== undefined
                                        ? `${ev.real.toFixed(3)} ${ev.imag >= 0 ? '+' : '-'} ${Math.abs(ev.imag).toFixed(3)}i`
                                        : ev.toFixed(3)}
                                    </div>
                                  ))
                                ) : (
                                  <div className="font-mono text-xs">λ = {eigenvalues.toFixed(3)}</div>
                                )}
                              </div>
                              <div className="mt-2 pt-2 border-t border-indigo-200">
                                <div className="text-xs text-gray-600">
                                  Product of eigenvalues: {productOfEigenvalues.toFixed(6)}
                                </div>
                                <div className="text-xs text-gray-600">
                                  Determinant: {det.toFixed(6)}
                                </div>
                                <div className={`text-xs font-semibold mt-1 ${
                                  Math.abs(productOfEigenvalues - det) < 0.01 
                                    ? 'text-green-700' 
                                    : 'text-orange-700'
                                }`}>
                                  {Math.abs(productOfEigenvalues - det) < 0.01 
                                    ? '✓ Match! (within rounding error)' 
                                    : 'Note: Complex eigenvalues use |λ|²'}
                                </div>
                              </div>
                            </div>
                            <div className="text-xs text-indigo-800">
                              <strong>Key Insight:</strong> If det(A) = 0, then at least one eigenvalue is 0, 
                              meaning the matrix has a zero eigenvalue and is singular.
                            </div>
                          </div>
                        </div>
                      );
                    }
                  } catch (e) {
                    return null;
                  }
                })()}

                {/* Connection to Matrix Invertibility */}
                <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
                  <h5 className="font-semibold text-green-900 mb-3">Connection to Matrix Invertibility:</h5>
                  <div className="space-y-3">
                    <div className={`bg-white rounded p-3 border-2 ${
                      Math.abs(det) < 1e-10 
                        ? 'border-red-300 bg-red-50' 
                        : 'border-green-300 bg-green-50'
                    }`}>
                      <div className="font-mono text-sm mb-2 font-semibold">
                        A is invertible ⟺ det(A) ≠ 0
                      </div>
                      <div className="text-xs text-gray-700 mb-2">
                        A matrix is invertible if and only if its determinant is non-zero
                      </div>
                      {Math.abs(det) < 1e-10 ? (
                        <div className="text-red-800">
                          <div className="font-semibold mb-1">⚠️ This matrix is NOT invertible</div>
                          <div className="text-xs">
                            • det(A) = {det.toFixed(6)} ≈ 0
                          </div>
                          <div className="text-xs">
                            • Matrix is singular (columns are linearly dependent)
                          </div>
                          <div className="text-xs">
                            • Cannot compute A⁻¹
                          </div>
                        </div>
                      ) : (
                        <div className="text-green-800">
                          <div className="font-semibold mb-1">✓ This matrix IS invertible</div>
                          <div className="text-xs">
                            • det(A) = {det.toFixed(6)} ≠ 0
                          </div>
                          <div className="text-xs">
                            • Columns are linearly independent
                          </div>
                          <div className="text-xs">
                            • A⁻¹ exists and can be computed
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="bg-white rounded p-3 border border-green-300">
                      <div className="text-xs text-gray-700 space-y-1">
                        <div><strong>Why this matters in ML:</strong></div>
                        <div>• Invertible matrices are needed for solving linear systems</div>
                        <div>• Non-invertible matrices cause numerical instability</div>
                        <div>• Regularization (e.g., L2) ensures matrices stay invertible</div>
                        <div>• Covariance matrices must be invertible for multivariate analysis</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {selectedTopic === 'matrix-operations' && (
        <div className="space-y-6">
          <div className="bg-indigo-50 border-2 border-indigo-200 rounded-lg p-4">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Matrix Operation Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setMatrixOperationType('multiplication')}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  matrixOperationType === 'multiplication'
                    ? 'bg-indigo-600 text-white shadow-lg'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                Multiplication
              </button>
              <button
                onClick={() => setMatrixOperationType('transformations')}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  matrixOperationType === 'transformations'
                    ? 'bg-indigo-600 text-white shadow-lg'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                Transformations
              </button>
            </div>
          </div>

          {matrixOperationType === 'multiplication' && (
            <>
              <h3 className="text-lg font-semibold text-gray-800">Matrix Multiplication</h3>
              
              {/* Animated Matrix Display */}
              <AnimatedMatrixDisplay
                matrixA={matrixA}
                matrixB={matrixB}
                matrixSizeA={matrixSizeA}
                matrixSizeB={matrixSizeB}
                highlightedRow={highlightedRow}
                highlightedCol={highlightedCol}
                activeStep={activeStep}
              />

              {/* Result Matrix */}
              {matrixSizeA.cols === matrixSizeB.rows && (
                <MatrixMultiplicationResult 
                  matrixA={matrixA} 
                  matrixB={matrixB}
                  matrixSizeA={matrixSizeA}
                  matrixSizeB={matrixSizeB}
                  highlightedRow={highlightedRow}
                  highlightedCol={highlightedCol}
                  activeStep={activeStep}
                  setHighlightedRow={setHighlightedRow}
                  setHighlightedCol={setHighlightedCol}
                  setActiveStep={setActiveStep}
                />
              )}

              {/* ML Application Info */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 mb-3">ML Applications</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded border border-gray-200">
                    <h5 className="font-semibold text-sm mb-1">Forward Propagation</h5>
                    <div className="font-mono text-xs">
                      output = activation(W × input + b)
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Each layer multiplies weights by input to produce output
                    </p>
                  </div>
                  <div className="bg-white p-3 rounded border border-gray-200">
                    <h5 className="font-semibold text-sm mb-1">Batch Processing</h5>
                    <div className="font-mono text-xs">
                      Batch × Weights = Output
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      Process multiple samples simultaneously using matrix multiplication
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}

          {matrixOperationType === 'transformations' && (
            <>
              <h3 className="text-lg font-semibold text-gray-800">Matrix Transformations</h3>
              <InteractiveMatrixTransformationsVisualization />
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Component for showing matrix multiplication result with step-by-step
function MatrixMultiplicationResult({ 
  matrixA, 
  matrixB, 
  matrixSizeA, 
  matrixSizeB,
  highlightedRow,
  highlightedCol,
  activeStep,
  setHighlightedRow,
  setHighlightedCol,
  setActiveStep
}) {
  const result = useMemo(() => {
    try {
      return math.multiplyMatrices(matrixA, matrixB);
    } catch (e) {
      return null;
    }
  }, [matrixA, matrixB]);

  const [selectedCell, setSelectedCell] = useState(null);
  const [showVisualGuide, setShowVisualGuide] = useState(false);

  // Update highlights when cell is selected and auto-show guide
  useEffect(() => {
    if (selectedCell) {
      setHighlightedRow(selectedCell.row);
      setHighlightedCol(selectedCell.col);
      setActiveStep(null);
      // Auto-show guide when cell is selected
      setShowVisualGuide(true);
    } else {
      setHighlightedRow(null);
      setHighlightedCol(null);
      setActiveStep(null);
    }
  }, [selectedCell, setHighlightedRow, setHighlightedCol, setActiveStep]);

  if (!result) return null;

  const getCalculationSteps = (row, col) => {
    const steps = [];
    for (let k = 0; k < matrixSizeA.cols; k++) {
      steps.push({
        a: matrixA[row][k],
        b: matrixB[k][col],
        product: matrixA[row][k] * matrixB[k][col],
        index: k
      });
    }
    return steps;
  };

  return (
    <div className="space-y-4">
      {/* Result Matrix - Show first so user can click */}
      <div className="bg-green-50 rounded-lg p-4">
        <div className="text-sm font-semibold text-green-900 mb-2">
          Result: {matrixSizeA.rows}×{matrixSizeB.cols} Matrix
        </div>
        <div className="font-mono text-sm">
          {result.map((row, i) => (
            <div key={i} className="flex gap-2 justify-center">
              {row.map((val, j) => (
                <button
                  key={j}
                  onClick={() => setSelectedCell(selectedCell?.row === i && selectedCell?.col === j ? null : { row: i, col: j })}
                  className={`w-16 text-center px-2 py-1 rounded transition-colors ${
                    selectedCell?.row === i && selectedCell?.col === j
                      ? 'bg-green-600 text-white font-bold ring-2 ring-green-800'
                      : 'bg-white hover:bg-green-100'
                  }`}
                >
                  {val.toFixed(1)}
                </button>
              ))}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-600 mt-2 text-center">
          💡 Click any cell above to see step-by-step calculation
        </p>
      </div>

      {/* Visual Guide Toggle */}
      <div className="flex items-center justify-between bg-indigo-50 rounded-lg p-3">
        <div>
          <h4 className="font-semibold text-indigo-900 text-sm">Visual Guide</h4>
          <p className="text-xs text-indigo-700">
            {selectedCell 
              ? `Showing calculation for Result[${selectedCell.row}][${selectedCell.col}]`
              : 'Click a result cell above to see the visual guide'}
          </p>
        </div>
        {selectedCell && (
          <div className="flex gap-2">
            <button
              onClick={() => setShowVisualGuide(!showVisualGuide)}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                showVisualGuide
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-indigo-600 border-2 border-indigo-300'
              }`}
            >
              {showVisualGuide ? 'Hide Guide' : 'Show Guide'}
            </button>
          </div>
        )}
      </div>

      {/* Prominent Animation Button - appears when cell is selected */}
      {selectedCell && showVisualGuide && (
        <div className="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h5 className="font-bold text-yellow-900 text-sm mb-1">
                🎬 Watch the Animation!
              </h5>
              <p className="text-xs text-yellow-800">
                Click the button below to see how row and column elements match step-by-step
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Visual Matrix Multiplication Guide */}
      {showVisualGuide && selectedCell && (
        <VisualMultiplicationGuide
          matrixA={matrixA}
          matrixB={matrixB}
          selectedCell={selectedCell}
          matrixSizeA={matrixSizeA}
          matrixSizeB={matrixSizeB}
          onStepChange={(step) => setActiveStep(step)}
        />
      )}

      {/* Message when no cell selected */}
      {!selectedCell && (
        <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4">
          <div className="text-sm text-yellow-900 text-center">
            💡 <strong>Tip:</strong> Click any cell in the result matrix above to see a detailed visual guide 
            showing how that value is calculated step-by-step!
          </div>
        </div>
      )}

      {/* Step-by-Step Calculation */}
      {selectedCell && (
        <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
          <h4 className="font-semibold text-blue-900 mb-3">
            Calculating Result[{selectedCell.row}][{selectedCell.col}]
          </h4>
          
          {/* Visual Row-Column Guide */}
          <div className="mb-4 bg-white rounded-lg p-3 border border-blue-200">
            <div className="text-xs font-semibold text-gray-700 mb-2">
              Step: Multiply Row {selectedCell.row} of A with Column {selectedCell.col} of B
            </div>
            <div className="flex items-center gap-4 text-xs">
              <div className="flex-1">
                <div className="text-gray-600 mb-1">Row {selectedCell.row} from A:</div>
                <div className="flex gap-1">
                  {matrixA[selectedCell.row].map((val, idx) => (
                    <div
                      key={idx}
                      className="bg-indigo-100 px-2 py-1 rounded font-mono font-semibold text-indigo-900"
                    >
                      {val.toFixed(1)}
                    </div>
                  ))}
                </div>
              </div>
              <div className="text-2xl text-gray-400">×</div>
              <div className="flex-1">
                <div className="text-gray-600 mb-1">Column {selectedCell.col} from B:</div>
                <div className="flex flex-col gap-1">
                  {matrixB.map((row, idx) => (
                    <div
                      key={idx}
                      className="bg-purple-100 px-2 py-1 rounded font-mono font-semibold text-purple-900 text-center"
                    >
                      {row[selectedCell.col].toFixed(1)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            {getCalculationSteps(selectedCell.row, selectedCell.col).map((step, idx) => (
              <div key={idx} className="bg-white rounded p-2 flex items-center gap-2">
                <span className="text-sm font-mono">
                  A[{selectedCell.row}][{step.index}] × B[{step.index}][{selectedCell.col}]
                </span>
                <span className="text-gray-400">=</span>
                <span className="text-sm font-mono">
                  {step.a.toFixed(1)} × {step.b.toFixed(1)}
                </span>
                <span className="text-gray-400">=</span>
                <span className="text-sm font-mono font-semibold text-blue-600">
                  {step.product.toFixed(2)}
                </span>
              </div>
            ))}
            <div className="bg-blue-100 rounded p-2 mt-2">
              <div className="text-sm font-semibold text-blue-900">
                Sum = {getCalculationSteps(selectedCell.row, selectedCell.col)
                  .reduce((sum, step) => sum + step.product, 0)
                  .toFixed(2)}
              </div>
            </div>
            <div className="text-xs text-gray-600 mt-2">
              Formula: C[i][j] = Σ(k=0 to {matrixSizeA.cols - 1}) A[i][k] × B[k][j]
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// Animated matrix display component
function AnimatedMatrixDisplay({ matrixA, matrixB, matrixSizeA, matrixSizeB, highlightedRow, highlightedCol, activeStep }) {
  const [hoveredCellA, setHoveredCellA] = useState(null);
  const [hoveredCellB, setHoveredCellB] = useState(null);

  return (
    <div className="space-y-4">
      {/* Matrix Display with Animation */}
      <div className="grid md:grid-cols-3 gap-4 items-center">
        {/* Matrix A */}
        <div className="bg-indigo-50 rounded-lg p-4">
          <div className="text-sm font-semibold text-indigo-900 mb-2">Matrix A</div>
          <div className="font-mono text-sm">
            {matrixA.map((row, i) => (
              <div
                key={i}
                className={`flex gap-2 justify-center mb-1 transition-all duration-300 ${
                  highlightedRow === i
                    ? 'bg-indigo-200 rounded px-2 py-1 transform scale-105'
                    : ''
                }`}
              >
                {row.map((val, j) => {
                  const cellKey = `A-${i}-${j}`;
                  const isHovered = hoveredCellA === cellKey;
                  return (
                    <span
                      key={j}
                      onMouseEnter={() => setHoveredCellA(cellKey)}
                      onMouseLeave={() => setHoveredCellA(null)}
                      className={`w-12 text-center px-2 py-1 rounded transition-all duration-300 cursor-pointer ${
                        highlightedRow === i && activeStep === j
                          ? 'bg-indigo-600 text-white font-bold ring-2 ring-yellow-400 animate-pulse'
                          : highlightedRow === i
                          ? 'bg-indigo-400 text-white font-semibold'
                          : isHovered
                          ? 'bg-indigo-200 text-indigo-900 font-semibold ring-2 ring-indigo-400 scale-110'
                          : 'bg-white hover:bg-indigo-50'
                      }`}
                      title={`A[${i}][${j}] = ${val.toFixed(2)}`}
                    >
                      {val.toFixed(1)}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
          {hoveredCellA && (
            <div className="text-xs text-indigo-700 mt-2 text-center">
              A[{hoveredCellA.split('-')[1]}][{hoveredCellA.split('-')[2]}] = {matrixA[parseInt(hoveredCellA.split('-')[1])][parseInt(hoveredCellA.split('-')[2])].toFixed(2)}
            </div>
          )}
        </div>

        {/* Multiplication Symbol */}
        <div className="text-center text-3xl font-bold text-gray-400">×</div>

        {/* Matrix B */}
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-sm font-semibold text-purple-900 mb-2">Matrix B</div>
          <div className="font-mono text-sm">
            {matrixB.map((row, i) => (
              <div key={i} className="flex gap-2 justify-center mb-1">
                {row.map((val, j) => {
                  const cellKey = `B-${i}-${j}`;
                  const isHovered = hoveredCellB === cellKey;
                  return (
                    <span
                      key={j}
                      onMouseEnter={() => setHoveredCellB(cellKey)}
                      onMouseLeave={() => setHoveredCellB(null)}
                      className={`w-12 text-center px-2 py-1 rounded transition-all duration-300 cursor-pointer ${
                        highlightedCol === j && activeStep === i
                          ? 'bg-purple-600 text-white font-bold ring-2 ring-yellow-400 animate-pulse'
                          : highlightedCol === j
                          ? 'bg-purple-400 text-white font-semibold'
                          : isHovered
                          ? 'bg-purple-200 text-purple-900 font-semibold ring-2 ring-purple-400 scale-110'
                          : 'bg-white hover:bg-purple-50'
                      }`}
                      title={`B[${i}][${j}] = ${val.toFixed(2)}`}
                    >
                      {val.toFixed(1)}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
          {hoveredCellB && (
            <div className="text-xs text-purple-700 mt-2 text-center">
              B[{hoveredCellB.split('-')[1]}][{hoveredCellB.split('-')[2]}] = {matrixB[parseInt(hoveredCellB.split('-')[1])][parseInt(hoveredCellB.split('-')[2])].toFixed(2)}
            </div>
          )}
        </div>
      </div>

      {/* Animation Controls */}
      <div className="bg-gray-50 rounded-lg p-3">
        <div className="text-xs text-gray-600 mb-2 text-center">
          Click a result cell below to see animated row-column matching
        </div>
      </div>
    </div>
  );
}

// Visual guide component showing row-column interaction
function VisualMultiplicationGuide({ matrixA, matrixB, selectedCell, matrixSizeA, matrixSizeB, onStepChange }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  if (!selectedCell) return null;

  const steps = matrixA[selectedCell.row].map((val, idx) => ({
    index: idx,
    aVal: val,
    bVal: matrixB[idx][selectedCell.col],
    product: val * matrixB[idx][selectedCell.col]
  }));

  const startAnimation = () => {
    setIsAnimating(true);
    setCurrentStep(0);
    if (onStepChange) onStepChange(0);
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        const nextStep = prev >= steps.length - 1 ? steps.length - 1 : prev + 1;
        if (onStepChange) onStepChange(nextStep);
        
        if (prev >= steps.length - 1) {
          clearInterval(interval);
          setIsAnimating(false);
          if (onStepChange) onStepChange(null);
          return steps.length - 1;
        }
        return nextStep;
      });
    }, 1000);
  };

  return (
    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
      <div className="flex flex-col sm:flex-row items-center justify-between mb-4 gap-3">
        <h4 className="font-bold text-indigo-900 text-center flex-1">
          Visual Guide: How Row × Column Works
        </h4>
        <button
          onClick={startAnimation}
          disabled={isAnimating}
          className={`px-6 py-3 rounded-lg text-base font-bold transition-all shadow-lg ${
            isAnimating
              ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-xl transform hover:scale-105'
          }`}
        >
          {isAnimating ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin">⏳</span>
              Animating...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <span>▶</span>
              Animate Row-Column Matching
            </span>
          )}
        </button>
      </div>
      
      <div className="grid md:grid-cols-2 gap-6">
        {/* Matrix A with animated highlighted row */}
        <div>
          <div className="text-sm font-semibold text-indigo-900 mb-2 text-center">
            Matrix A - Row {selectedCell.row} highlighted
          </div>
          <div className="bg-white rounded-lg p-3 border-2 border-indigo-300">
            {matrixA.map((row, i) => (
              <div
                key={i}
                className={`flex gap-2 justify-center mb-1 transition-all duration-500 ${
                  i === selectedCell.row ? 'bg-indigo-200 rounded px-2 py-1 transform scale-105' : ''
                }`}
              >
                {row.map((val, j) => (
                  <div
                    key={j}
                    className={`w-12 text-center px-2 py-1 rounded font-mono text-sm transition-all duration-500 ${
                      i === selectedCell.row && isAnimating && currentStep === j
                        ? 'bg-yellow-400 text-gray-900 font-bold ring-4 ring-yellow-500 scale-110 animate-bounce'
                        : i === selectedCell.row && isAnimating && currentStep > j
                        ? 'bg-green-400 text-white font-bold'
                        : i === selectedCell.row
                        ? 'bg-indigo-500 text-white font-bold'
                        : 'bg-gray-100'
                    }`}
                  >
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="text-xs text-gray-600 mt-2 text-center">
            ↑ This entire row is used
          </div>
        </div>

        {/* Matrix B with animated highlighted column */}
        <div>
          <div className="text-sm font-semibold text-purple-900 mb-2 text-center">
            Matrix B - Column {selectedCell.col} highlighted
          </div>
          <div className="bg-white rounded-lg p-3 border-2 border-purple-300">
            {matrixB.map((row, i) => (
              <div key={i} className="flex gap-2 justify-center mb-1">
                {row.map((val, j) => (
                  <div
                    key={j}
                    className={`w-12 text-center px-2 py-1 rounded font-mono text-sm transition-all duration-500 ${
                      j === selectedCell.col && isAnimating && currentStep === i
                        ? 'bg-yellow-400 text-gray-900 font-bold ring-4 ring-yellow-500 scale-110 animate-bounce'
                        : j === selectedCell.col && isAnimating && currentStep > i
                        ? 'bg-green-400 text-white font-bold'
                        : j === selectedCell.col
                        ? 'bg-purple-500 text-white font-bold'
                        : 'bg-gray-100'
                    }`}
                  >
                    {val.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="text-xs text-gray-600 mt-2 text-center">
            ↑ This entire column is used
          </div>
        </div>
      </div>

      {/* Animated Multiplication visualization */}
      <div className="mt-6 bg-white rounded-lg p-4 border-2 border-green-300">
        <div className="text-sm font-semibold text-green-900 mb-3 text-center">
          Multiplication Process {isAnimating && `(Step ${currentStep + 1}/${steps.length})`}
        </div>
        <div className="space-y-2">
          {steps.map((step, idx) => (
            <div
              key={idx}
              className={`flex items-center justify-center gap-3 rounded p-2 transition-all duration-500 ${
                isAnimating && currentStep === idx
                  ? 'bg-yellow-100 border-2 border-yellow-500 transform scale-105'
                  : isAnimating && currentStep > idx
                  ? 'bg-green-100 border border-green-300'
                  : 'bg-green-50'
              }`}
            >
              <div className={`px-3 py-1 rounded font-mono font-bold transition-all duration-500 ${
                isAnimating && currentStep === idx
                  ? 'bg-yellow-500 text-gray-900 ring-2 ring-yellow-600'
                  : isAnimating && currentStep > idx
                  ? 'bg-green-500 text-white'
                  : 'bg-indigo-500 text-white'
              }`}>
                {step.aVal.toFixed(1)}
              </div>
              <span className="text-gray-400 font-bold">×</span>
              <div className={`px-3 py-1 rounded font-mono font-bold transition-all duration-500 ${
                isAnimating && currentStep === idx
                  ? 'bg-yellow-500 text-gray-900 ring-2 ring-yellow-600'
                  : isAnimating && currentStep > idx
                  ? 'bg-green-500 text-white'
                  : 'bg-purple-500 text-white'
              }`}>
                {step.bVal.toFixed(1)}
              </div>
              <span className="text-gray-400">=</span>
              <div className={`px-3 py-1 rounded font-mono font-bold transition-all duration-500 ${
                isAnimating && currentStep === idx
                  ? 'bg-yellow-600 text-white ring-2 ring-yellow-700 animate-pulse'
                  : isAnimating && currentStep > idx
                  ? 'bg-green-600 text-white'
                  : 'bg-green-500 text-white'
              }`}>
                {step.product.toFixed(2)}
              </div>
            </div>
          ))}
          <div className="border-t-2 border-green-400 mt-3 pt-2">
            <div className="flex items-center justify-center gap-2">
              <span className="text-sm font-semibold text-gray-700">Sum =</span>
              <div className="bg-green-600 text-white px-4 py-2 rounded font-mono font-bold text-lg">
                {matrixA[selectedCell.row]
                  .reduce(
                    (sum, val, idx) =>
                      sum + val * matrixB[idx][selectedCell.col],
                    0
                  )
                  .toFixed(2)}
              </div>
            </div>
            <div className="text-xs text-gray-600 mt-2 text-center">
              This sum becomes Result[{selectedCell.row}][{selectedCell.col}]
            </div>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 bg-yellow-50 rounded-lg p-3 border border-yellow-200">
        <div className="text-xs text-yellow-900">
          <strong>How it works:</strong> Take each element from Row {selectedCell.row} of A, 
          multiply it with the corresponding element from Column {selectedCell.col} of B, 
          then sum all the products together.
        </div>
      </div>
    </div>
  );
}
