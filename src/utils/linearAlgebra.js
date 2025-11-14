// Linear Algebra utilities for AI/ML applications

import * as math from './math.js';

/**
 * Calculate eigenvalues and eigenvectors for a matrix
 */
export function calculateEigenDecomposition(matrix) {
  if (matrix.length !== 2 || matrix[0].length !== 2) {
    throw new Error('Currently only supports 2x2 matrices');
  }
  
  const eigenvalues = math.eigenvalues2x2(matrix);
  const eigenvectors = [];
  
  if (Array.isArray(eigenvalues) && eigenvalues.length > 0) {
    if (typeof eigenvalues[0] === 'object' && eigenvalues[0].imag !== undefined) {
      // Complex eigenvalues
      return {
        eigenvalues,
        eigenvectors: null,
        isComplex: true
      };
    }
    
    eigenvalues.forEach(eigenvalue => {
      const eigenvector = math.eigenvector2x2(matrix, eigenvalue);
      eigenvectors.push({
        eigenvalue,
        eigenvector
      });
    });
  }
  
  return {
    eigenvalues,
    eigenvectors,
    isComplex: false
  };
}

/**
 * Represent data as a matrix (for ML applications)
 */
export function dataToMatrix(dataPoints) {
  // Each data point becomes a row
  return dataPoints.map(point => {
    if (Array.isArray(point)) {
      return point;
    }
    return [point.x || point[0], point.y || point[1]];
  });
}

/**
 * Represent weights as a matrix (for neural networks)
 */
export function weightsToMatrix(weights, inputSize, outputSize) {
  // Reshape flat array into matrix
  const matrix = [];
  for (let i = 0; i < outputSize; i++) {
    const row = [];
    for (let j = 0; j < inputSize; j++) {
      const index = i * inputSize + j;
      row.push(weights[index] || 0);
    }
    matrix.push(row);
  }
  return matrix;
}

/**
 * Matrix-vector multiplication (common in ML)
 */
export function matrixVectorMultiply(matrix, vector) {
  const result = [];
  for (let i = 0; i < matrix.length; i++) {
    let sum = 0;
    for (let j = 0; j < vector.length; j++) {
      sum += matrix[i][j] * vector[j];
    }
    result.push(sum);
  }
  return result;
}

/**
 * Apply transformation to data points
 */
export function transformData(dataPoints, transformationMatrix) {
  const matrix = dataToMatrix(dataPoints);
  return matrix.map(row => {
    const result = math.multiplyMatrices([row], transformationMatrix)[0];
    return { x: result[0], y: result[1] };
  });
}

