// Supervised Learning utilities for AI/ML applications

import * as math from './math.js';

/**
 * Calculate Mean Squared Error (MSE)
 */
export function meanSquaredError(predictions, actuals) {
  if (predictions.length !== actuals.length) {
    throw new Error('Predictions and actuals must have the same length');
  }
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.pow(predictions[i] - actuals[i], 2);
  }
  return sum / n;
}

/**
 * Calculate Mean Absolute Error (MAE)
 */
export function meanAbsoluteError(predictions, actuals) {
  if (predictions.length !== actuals.length) {
    throw new Error('Predictions and actuals must have the same length');
  }
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.abs(predictions[i] - actuals[i]);
  }
  return sum / n;
}

/**
 * Calculate Cross-Entropy Loss (Binary)
 */
export function binaryCrossEntropy(predictions, actuals, epsilon = 1e-15) {
  if (predictions.length !== actuals.length) {
    throw new Error('Predictions and actuals must have the same length');
  }
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.max(epsilon, Math.min(1 - epsilon, predictions[i]));
    sum += -(actuals[i] * Math.log(p) + (1 - actuals[i]) * Math.log(1 - p));
  }
  return sum / n;
}

/**
 * Calculate Hinge Loss
 */
export function hingeLoss(predictions, actuals) {
  if (predictions.length !== actuals.length) {
    throw new Error('Predictions and actuals must have the same length');
  }
  const n = predictions.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.max(0, 1 - actuals[i] * predictions[i]);
  }
  return sum / n;
}

/**
 * Calculate confusion matrix
 */
export function confusionMatrix(predictions, actuals, threshold = 0.5) {
  const binaryPreds = predictions.map(p => p >= threshold ? 1 : 0);
  let tp = 0, tn = 0, fp = 0, fn = 0;
  
  for (let i = 0; i < binaryPreds.length; i++) {
    if (binaryPreds[i] === 1 && actuals[i] === 1) tp++;
    else if (binaryPreds[i] === 0 && actuals[i] === 0) tn++;
    else if (binaryPreds[i] === 1 && actuals[i] === 0) fp++;
    else if (binaryPreds[i] === 0 && actuals[i] === 1) fn++;
  }
  
  return { tp, tn, fp, fn };
}

/**
 * Calculate accuracy
 */
export function accuracy(predictions, actuals, threshold = 0.5) {
  const cm = confusionMatrix(predictions, actuals, threshold);
  const total = cm.tp + cm.tn + cm.fp + cm.fn;
  return total > 0 ? (cm.tp + cm.tn) / total : 0;
}

/**
 * Calculate precision
 */
export function precision(predictions, actuals, threshold = 0.5) {
  const cm = confusionMatrix(predictions, actuals, threshold);
  return cm.tp + cm.fp > 0 ? cm.tp / (cm.tp + cm.fp) : 0;
}

/**
 * Calculate recall (sensitivity)
 */
export function recall(predictions, actuals, threshold = 0.5) {
  const cm = confusionMatrix(predictions, actuals, threshold);
  return cm.tp + cm.fn > 0 ? cm.tp / (cm.tp + cm.fn) : 0;
}

/**
 * Calculate F1-score
 */
export function f1Score(predictions, actuals, threshold = 0.5) {
  const prec = precision(predictions, actuals, threshold);
  const rec = recall(predictions, actuals, threshold);
  return prec + rec > 0 ? 2 * (prec * rec) / (prec + rec) : 0;
}

/**
 * Calculate ROC curve points
 */
export function rocCurve(predictions, actuals, numPoints = 100) {
  const thresholds = [];
  for (let i = 0; i <= numPoints; i++) {
    thresholds.push(i / numPoints);
  }
  
  const rocPoints = thresholds.map(threshold => {
    const cm = confusionMatrix(predictions, actuals, threshold);
    const tpr = cm.tp + cm.fn > 0 ? cm.tp / (cm.tp + cm.fn) : 0;
    const fpr = cm.fp + cm.tn > 0 ? cm.fp / (cm.fp + cm.tn) : 0;
    return { threshold, tpr, fpr };
  });
  
  return rocPoints;
}

/**
 * Calculate AUC (Area Under Curve) using trapezoidal rule
 */
export function auc(rocPoints) {
  let area = 0;
  for (let i = 1; i < rocPoints.length; i++) {
    const width = rocPoints[i].fpr - rocPoints[i - 1].fpr;
    const avgHeight = (rocPoints[i].tpr + rocPoints[i - 1].tpr) / 2;
    area += width * avgHeight;
  }
  return area;
}

/**
 * L1 Regularization (Lasso)
 */
export function l1Regularization(weights, lambda) {
  return lambda * weights.reduce((sum, w) => sum + Math.abs(w), 0);
}

/**
 * L2 Regularization (Ridge)
 */
export function l2Regularization(weights, lambda) {
  return lambda * weights.reduce((sum, w) => sum + w * w, 0);
}

/**
 * Generate sample data for bias-variance demonstration
 */
export function generateBiasVarianceData(n = 50, noise = 0.1) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const x = (i / n) * 4 - 2; // -2 to 2
    const trueY = 0.5 * x * x + 0.3 * x; // True function
    const noisyY = trueY + (Math.random() - 0.5) * noise * 2;
    data.push({ x, y: noisyY, trueY });
  }
  return data;
}

/**
 * Fit polynomial of given degree
 */
export function fitPolynomial(data, degree) {
  // Simple polynomial fitting using least squares
  // This is a simplified version - in practice, use proper matrix operations
  const n = data.length;
  const X = data.map(d => {
    const row = [];
    for (let i = 0; i <= degree; i++) {
      row.push(Math.pow(d.x, i));
    }
    return row;
  });
  
  const y = data.map(d => d.y);
  
  // Simplified: return coefficients for visualization
  // In practice, solve: (X^T * X)^(-1) * X^T * y
  return { degree, data };
}

