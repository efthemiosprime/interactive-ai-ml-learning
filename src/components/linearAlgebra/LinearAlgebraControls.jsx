import React, { useMemo } from 'react';
import { RefreshCw } from 'lucide-react';

export default function LinearAlgebraControls({ 
  selectedTopic, 
  setSelectedTopic, 
  matrix, 
  setMatrix,
  matrixA,
  setMatrixA,
  matrixB,
  setMatrixB,
  matrixSizeA,
  setMatrixSizeA,
  matrixSizeB,
  setMatrixSizeB
}) {
  const topics = [
    { id: 'eigenvalues', label: 'Eigenvalues & Eigenvectors' },
    { id: 'data-representation', label: 'Data Representation' },
    { id: 'weight-representation', label: 'Weight Representation' },
    { id: 'matrix-operations', label: 'Matrix Operations' },
    { id: 'determinant', label: 'Determinant' }
  ];

  const matrixSizes = [
    { rows: 2, cols: 2, label: '2×2' },
    { rows: 2, cols: 3, label: '2×3' },
    { rows: 3, cols: 2, label: '3×2' },
    { rows: 3, cols: 3, label: '3×3' },
    { rows: 3, cols: 4, label: '3×4' },
    { rows: 4, cols: 3, label: '4×3' },
    { rows: 4, cols: 4, label: '4×4' },
  ];

  const sampleMatrices = {
    '2x2': [
      { A: [[1, 2], [3, 4]], B: [[5, 6], [7, 8]] },
      { A: [[2, 1], [1, 2]], B: [[1, 0], [0, 1]] },
      { A: [[1, 0], [0, 1]], B: [[3, 4], [5, 6]] },
      { A: [[2, 3], [1, 4]], B: [[1, 2], [3, 1]] },
    ],
    '3x3': [
      { A: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], B: [[9, 8, 7], [6, 5, 4], [3, 2, 1]] },
      { A: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], B: [[2, 3, 4], [5, 6, 7], [8, 9, 10]] },
      { A: [[1, 2, 1], [2, 1, 2], [1, 2, 1]], B: [[2, 1, 2], [1, 2, 1], [2, 1, 2]] },
    ],
    '2x3_3x2': [
      { A: [[1, 2, 3], [4, 5, 6]], B: [[7, 8], [9, 10], [11, 12]] },
      { A: [[2, 1, 3], [1, 2, 1]], B: [[1, 2], [3, 4], [5, 6]] },
      { A: [[1, 0, 1], [0, 1, 0]], B: [[2, 3], [4, 5], [6, 7]] },
    ],
    '2x3_3x3': [
      { A: [[1, 2, 3], [4, 5, 6]], B: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] },
      { A: [[2, 1, 3], [1, 2, 1]], B: [[1, 0, 1], [0, 1, 0], [1, 1, 1]] },
    ],
    '3x2_2x3': [
      { A: [[1, 2], [3, 4], [5, 6]], B: [[1, 2, 3], [4, 5, 6]] },
      { A: [[2, 1], [1, 2], [3, 1]], B: [[1, 2, 1], [2, 1, 2]] },
    ],
    '3x4_4x3': [
      { A: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], B: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] },
    ],
    '4x4': [
      { A: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], 
        B: [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]] },
      { A: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
        B: [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]] },
    ],
  };

  const updateMatrix = (row, col, value) => {
    const newMatrix = matrix.map((r, i) =>
      r.map((c, j) => (i === row && j === col ? parseFloat(value) || 0 : c))
    );
    setMatrix(newMatrix);
  };

  const createEmptyMatrix = (rows, cols) => {
    return Array(rows).fill(0).map(() => Array(cols).fill(0));
  };

  const updateMatrixA = (row, col, value) => {
    const newMatrix = matrixA.map((r, i) =>
      r.map((c, j) => (i === row && j === col ? parseFloat(value) || 0 : c))
    );
    setMatrixA(newMatrix);
  };

  const updateMatrixB = (row, col, value) => {
    const newMatrix = matrixB.map((r, i) =>
      r.map((c, j) => (i === row && j === col ? parseFloat(value) || 0 : c))
    );
    setMatrixB(newMatrix);
  };

  const handleSizeChangeA = (rows, cols) => {
    setMatrixSizeA({ rows, cols });
    setMatrixA(createEmptyMatrix(rows, cols));
  };

  const handleSizeChangeB = (rows, cols) => {
    setMatrixSizeB({ rows, cols });
    setMatrixB(createEmptyMatrix(rows, cols));
    // Auto-adjust A's columns to match B's rows for valid multiplication
    if (matrixSizeA.cols !== rows) {
      setMatrixSizeA({ ...matrixSizeA, cols: rows });
      const newA = createEmptyMatrix(matrixSizeA.rows, rows);
      // Copy existing values where possible
      for (let i = 0; i < Math.min(matrixSizeA.rows, matrixA.length); i++) {
        for (let j = 0; j < Math.min(rows, matrixA[i].length); j++) {
          newA[i][j] = matrixA[i][j];
        }
      }
      setMatrixA(newA);
    }
  };

  const loadSample = () => {
    // Try exact match first
    const exactKey = `${matrixSizeA.rows}x${matrixSizeA.cols}_${matrixSizeB.rows}x${matrixSizeB.cols}`;
    let samples = sampleMatrices[exactKey];
    
    // Try square matrix match
    if (!samples && matrixSizeA.rows === matrixSizeA.cols && matrixSizeB.rows === matrixSizeB.cols) {
      samples = sampleMatrices[`${matrixSizeA.rows}x${matrixSizeA.rows}`];
    }
    
    // Try size-based match
    if (!samples) {
      const sizeKey = `${matrixSizeA.rows}x${matrixSizeA.cols}`;
      samples = sampleMatrices[sizeKey];
    }
    
    if (samples && samples.length > 0) {
      // Randomly select from available samples
      const randomIndex = Math.floor(Math.random() * samples.length);
      const sample = samples[randomIndex];
      
      // Ensure dimensions match
      if (sample.A.length === matrixSizeA.rows && 
          sample.A[0].length === matrixSizeA.cols &&
          sample.B.length === matrixSizeB.rows && 
          sample.B[0].length === matrixSizeB.cols) {
        setMatrixA(sample.A);
        setMatrixB(sample.B);
      } else {
        // Create matrices with correct dimensions
        const newA = createEmptyMatrix(matrixSizeA.rows, matrixSizeA.cols);
        const newB = createEmptyMatrix(matrixSizeB.rows, matrixSizeB.cols);
        
        // Copy sample values where dimensions match
        for (let i = 0; i < Math.min(matrixSizeA.rows, sample.A.length); i++) {
          for (let j = 0; j < Math.min(matrixSizeA.cols, sample.A[i]?.length || 0); j++) {
            newA[i][j] = sample.A[i][j] || 0;
          }
        }
        
        for (let i = 0; i < Math.min(matrixSizeB.rows, sample.B.length); i++) {
          for (let j = 0; j < Math.min(matrixSizeB.cols, sample.B[i]?.length || 0); j++) {
            newB[i][j] = sample.B[i][j] || 0;
          }
        }
        
        setMatrixA(newA);
        setMatrixB(newB);
      }
    } else {
      // Generate simple sample matrices
      const newA = matrixA.map(row => row.map(() => Math.floor(Math.random() * 10) + 1));
      const newB = matrixB.map(row => row.map(() => Math.floor(Math.random() * 10) + 1));
      setMatrixA(newA);
      setMatrixB(newB);
    }
  };

  const canMultiply = useMemo(() => {
    return matrixSizeA.cols === matrixSizeB.rows;
  }, [matrixSizeA, matrixSizeB]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Controls</h2>

      {/* Topic Selector */}
      <div className="mb-6">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Topic
        </label>
        <select
          value={selectedTopic}
          onChange={(e) => setSelectedTopic(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        >
          {topics.map(topic => (
            <option key={topic.id} value={topic.id}>
              {topic.label}
            </option>
          ))}
        </select>
      </div>

      {/* Matrix Input for Eigenvalues */}
      {selectedTopic === 'eigenvalues' && (
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            2x2 Matrix
          </label>
          <div className="space-y-2">
            {matrix.map((row, i) => (
              <div key={i} className="flex gap-2">
                {row.map((val, j) => (
                  <input
                    key={j}
                    type="number"
                    value={val}
                    onChange={(e) => updateMatrix(i, j, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    step="0.1"
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Matrix Input for Determinant */}
      {selectedTopic === 'determinant' && (
        <div className="mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Square Matrix (2×2, 3×3, or 4×4)
          </label>
          <div className="mb-3">
            <select
              value={`${matrix.length}x${matrix[0]?.length || 0}`}
              onChange={(e) => {
                const size = parseInt(e.target.value);
                const newMatrix = Array(size).fill(0).map(() => Array(size).fill(0));
                // Initialize with identity matrix
                for (let i = 0; i < size; i++) {
                  newMatrix[i][i] = 1;
                }
                setMatrix(newMatrix);
              }}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 mb-2"
            >
              <option value="2x2">2×2</option>
              <option value="3x3">3×3</option>
              <option value="4x4">4×4</option>
            </select>
          </div>
          <div className="space-y-2">
            {matrix.map((row, i) => (
              <div key={i} className="flex gap-2">
                {row.map((val, j) => (
                  <input
                    key={j}
                    type="number"
                    value={val}
                    onChange={(e) => updateMatrix(i, j, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    step="0.1"
                  />
                ))}
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-600 mt-2">
            💡 Determinant is only defined for square matrices
          </p>
        </div>
      )}

      {/* Matrix Multiplication Controls */}
      {selectedTopic === 'matrix-operations' && (
        <div className="space-y-6">
          {/* Matrix A Size */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Matrix A Size
            </label>
            <select
              value={`${matrixSizeA.rows}x${matrixSizeA.cols}`}
              onChange={(e) => {
                const [rows, cols] = e.target.value.split('x').map(Number);
                handleSizeChangeA(rows, cols);
              }}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            >
              {matrixSizes.map(size => (
                <option key={`${size.rows}x${size.cols}`} value={`${size.rows}x${size.cols}`}>
                  {size.label}
                </option>
              ))}
            </select>
          </div>

          {/* Matrix A Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Matrix A ({matrixSizeA.rows}×{matrixSizeA.cols})
            </label>
            <div className="space-y-1">
              {matrixA.map((row, i) => (
                <div key={i} className="flex gap-1">
                  {row.map((val, j) => (
                    <input
                      key={j}
                      type="number"
                      value={val}
                      onChange={(e) => updateMatrixA(i, j, e.target.value)}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-indigo-500"
                      step="0.1"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Matrix B Size */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Matrix B Size
            </label>
            <select
              value={`${matrixSizeB.rows}x${matrixSizeB.cols}`}
              onChange={(e) => {
                const [rows, cols] = e.target.value.split('x').map(Number);
                handleSizeChangeB(rows, cols);
              }}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            >
              {matrixSizes.filter(size => size.rows === matrixSizeA.cols).map(size => (
                <option key={`${size.rows}x${size.cols}`} value={`${size.rows}x${size.cols}`}>
                  {size.label}
                </option>
              ))}
            </select>
            {!canMultiply && (
              <p className="text-xs text-red-600 mt-1">
                Matrix A columns ({matrixSizeA.cols}) must equal Matrix B rows ({matrixSizeB.rows})
              </p>
            )}
          </div>

          {/* Matrix B Input */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Matrix B ({matrixSizeB.rows}×{matrixSizeB.cols})
            </label>
            <div className="space-y-1">
              {matrixB.map((row, i) => (
                <div key={i} className="flex gap-1">
                  {row.map((val, j) => (
                    <input
                      key={j}
                      type="number"
                      value={val}
                      onChange={(e) => updateMatrixB(i, j, e.target.value)}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-indigo-500"
                      step="0.1"
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Sample Matrices Button */}
          <button
            onClick={loadSample}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span className="text-sm font-semibold">Load Sample Matrices</span>
          </button>
        </div>
      )}

      {/* Info Panel */}
      <div className="bg-indigo-50 rounded-lg p-4">
        <h3 className="font-semibold text-indigo-900 mb-2">ML Application</h3>
        <p className="text-sm text-indigo-800">
          {selectedTopic === 'eigenvalues' && 
            'Eigenvalues help understand data transformations and principal components in PCA.'}
          {selectedTopic === 'data-representation' && 
            'Data is represented as matrices where rows are samples and columns are features.'}
          {selectedTopic === 'weight-representation' && 
            'Neural network weights are stored as matrices connecting layers.'}
          {selectedTopic === 'matrix-operations' && 
            canMultiply 
              ? `Result will be ${matrixSizeA.rows}×${matrixSizeB.cols} matrix. Practice with different sizes!`
              : 'Adjust matrix sizes to enable multiplication. A columns must equal B rows.'}
          {selectedTopic === 'determinant' && 
            'Determinant measures matrix invertibility and volume scaling. Used in solving systems and transformations.'}
        </p>
      </div>
    </div>
  );
}

