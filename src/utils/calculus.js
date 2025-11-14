// Calculus utilities for AI/ML applications

import * as math from './math.js';

/**
 * Calculate gradient for a loss function
 */
export function calculateGradient(lossFunction, weights, h = 1e-5) {
  const gradients = [];
  
  for (let i = 0; i < weights.length; i++) {
    const weightsPlus = [...weights];
    const weightsMinus = [...weights];
    weightsPlus[i] += h;
    weightsMinus[i] -= h;
    
    const gradient = (lossFunction(weightsPlus) - lossFunction(weightsMinus)) / (2 * h);
    gradients.push(gradient);
  }
  
  return gradients;
}

/**
 * Gradient descent step
 */
export function gradientDescentStep(weights, gradients, learningRate) {
  return weights.map((w, i) => w - learningRate * gradients[i]);
}

/**
 * Chain rule for backpropagation
 * For a composite function f(g(x)), calculate derivative
 */
export function backpropagationChainRule(outerDerivative, innerDerivative, x) {
  return math.chainRule(
    () => outerDerivative,
    () => x,
    () => innerDerivative,
    x
  );
}

/**
 * Calculate partial derivatives for a multi-variable function
 */
export function calculatePartialDerivatives(f, variables, h = 1e-5) {
  const partials = {};
  
  Object.keys(variables).forEach(key => {
    const varsPlus = { ...variables };
    const varsMinus = { ...variables };
    varsPlus[key] += h;
    varsMinus[key] -= h;
    
    partials[key] = (f(varsPlus) - f(varsMinus)) / (2 * h);
  });
  
  return partials;
}

/**
 * Example loss functions for ML
 */
export const lossFunctions = {
  // Mean Squared Error
  mse: (predicted, actual) => {
    if (predicted.length !== actual.length) return 0;
    let sum = 0;
    for (let i = 0; i < predicted.length; i++) {
      sum += Math.pow(predicted[i] - actual[i], 2);
    }
    return sum / predicted.length;
  },
  
  // Cross-entropy loss
  crossEntropy: (predicted, actual) => {
    if (predicted.length !== actual.length) return 0;
    let sum = 0;
    for (let i = 0; i < predicted.length; i++) {
      const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted[i]));
      sum += actual[i] * Math.log(p) + (1 - actual[i]) * Math.log(1 - p);
    }
    return -sum / predicted.length;
  }
};

