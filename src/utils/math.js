// Core mathematical functions for AI/ML applications

// ========== Linear Algebra ==========

/**
 * Multiply two matrices
 */
export function multiplyMatrices(a, b) {
  const rowsA = a.length;
  const colsA = a[0].length;
  const rowsB = b.length;
  const colsB = b[0].length;
  
  if (colsA !== rowsB) {
    throw new Error('Matrices dimensions do not match for multiplication');
  }
  
  const result = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));
  
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  
  return result;
}

/**
 * Calculate determinant of a 2x2 matrix
 */
export function determinant2x2(matrix) {
  return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
}

/**
 * Calculate determinant of a 3x3 matrix
 */
export function determinant3x3(matrix) {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[0][2];
  const d = matrix[1][0];
  const e = matrix[1][1];
  const f = matrix[1][2];
  const g = matrix[2][0];
  const h = matrix[2][1];
  const i = matrix[2][2];
  
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/**
 * Calculate determinant of any square matrix using Laplace expansion
 */
export function determinant(matrix) {
  const n = matrix.length;
  
  if (n === 0) return 0;
  if (n === 1) return matrix[0][0];
  if (n === 2) return determinant2x2(matrix);
  if (n === 3) return determinant3x3(matrix);
  
  // For larger matrices, use Laplace expansion along first row
  let det = 0;
  for (let j = 0; j < n; j++) {
    const minor = [];
    for (let i = 1; i < n; i++) {
      const row = [];
      for (let k = 0; k < n; k++) {
        if (k !== j) row.push(matrix[i][k]);
      }
      minor.push(row);
    }
    const sign = j % 2 === 0 ? 1 : -1;
    det += sign * matrix[0][j] * determinant(minor);
  }
  
  return det;
}

/**
 * Calculate eigenvalues for a 2x2 matrix
 */
export function eigenvalues2x2(matrix) {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[1][0];
  const d = matrix[1][1];
  
  // Characteristic polynomial: λ² - (a+d)λ + (ad - bc) = 0
  const trace = a + d;
  const det = a * d - b * c;
  
  const discriminant = trace * trace - 4 * det;
  
  if (discriminant < 0) {
    // Complex eigenvalues
    const real = trace / 2;
    const imag = Math.sqrt(-discriminant) / 2;
    return [
      { real, imag: imag },
      { real, imag: -imag }
    ];
  }
  
  const sqrtDisc = Math.sqrt(discriminant);
  return [
    (trace + sqrtDisc) / 2,
    (trace - sqrtDisc) / 2
  ];
}

/**
 * Calculate eigenvectors for a 2x2 matrix given an eigenvalue
 */
export function eigenvector2x2(matrix, eigenvalue) {
  const a = matrix[0][0];
  const b = matrix[0][1];
  const c = matrix[1][0];
  const d = matrix[1][1];
  
  // Solve (A - λI)v = 0
  const aMinusLambda = a - eigenvalue;
  const dMinusLambda = d - eigenvalue;
  
  // If b is not zero, use first row
  if (Math.abs(b) > 1e-10) {
    return { x: b, y: -(aMinusLambda) };
  }
  
  // If c is not zero, use second row
  if (Math.abs(c) > 1e-10) {
    return { x: -(dMinusLambda), y: c };
  }
  
  // Default eigenvector
  return { x: 1, y: 0 };
}

/**
 * Transpose a matrix
 */
export function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result = Array(cols).fill(0).map(() => Array(rows).fill(0));
  
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  
  return result;
}

// ========== Calculus ==========

/**
 * Numerical derivative using finite differences
 */
export function derivative(f, x, h = 1e-5) {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * Partial derivative of a function f(x, y) with respect to x
 */
export function partialDerivativeX(f, x, y, h = 1e-5) {
  return (f(x + h, y) - f(x - h, y)) / (2 * h);
}

/**
 * Partial derivative of a function f(x, y) with respect to y
 */
export function partialDerivativeY(f, x, y, h = 1e-5) {
  return (f(x, y + h) - f(x, y - h)) / (2 * h);
}

/**
 * Calculate gradient of a function f(x, y)
 */
export function gradient(f, x, y, h = 1e-5) {
  return {
    x: partialDerivativeX(f, x, y, h),
    y: partialDerivativeY(f, x, y, h)
  };
}

/**
 * Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x)
 */
export function chainRule(fPrime, g, gPrime, x) {
  return fPrime(g(x)) * gPrime(x);
}

// ========== Statistics ==========

/**
 * Calculate mean of an array
 */
export function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate variance of an array
 */
export function variance(values) {
  if (values.length === 0) return 0;
  const m = mean(values);
  const squaredDiffs = values.map(val => Math.pow(val - m, 2));
  return mean(squaredDiffs);
}

/**
 * Calculate standard deviation
 */
export function standardDeviation(values) {
  return Math.sqrt(variance(values));
}

/**
 * Calculate covariance between two arrays
 */
export function covariance(x, y) {
  if (x.length !== y.length || x.length === 0) return 0;
  
  const meanX = mean(x);
  const meanY = mean(y);
  
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    sum += (x[i] - meanX) * (y[i] - meanY);
  }
  
  return sum / x.length;
}

/**
 * Calculate correlation coefficient
 */
export function correlation(x, y) {
  const cov = covariance(x, y);
  const stdX = standardDeviation(x);
  const stdY = standardDeviation(y);
  
  if (stdX === 0 || stdY === 0) return 0;
  
  return cov / (stdX * stdY);
}

// ========== Probability ==========

/**
 * Conditional probability P(A|B) = P(A and B) / P(B)
 */
export function conditionalProbability(pAAndB, pB) {
  if (pB === 0) return 0;
  return pAAndB / pB;
}

/**
 * Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
 */
export function bayesTheorem(pBGivenA, pA, pB) {
  if (pB === 0) return 0;
  return (pBGivenA * pA) / pB;
}

/**
 * Normal distribution PDF
 */
export function normalPDF(x, mean, stdDev) {
  const variance = stdDev * stdDev;
  const coefficient = 1 / (stdDev * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mean) / stdDev, 2);
  return coefficient * Math.exp(exponent);
}

/**
 * Bernoulli distribution PMF
 */
export function bernoulliPMF(x, p) {
  if (x === 1) return p;
  if (x === 0) return 1 - p;
  return 0;
}

