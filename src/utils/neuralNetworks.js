// Neural Networks utilities for AI/ML applications

/**
 * Apply activation function to a value
 */
export function applyActivation(x, activationType) {
  switch (activationType) {
    case 'relu':
      return Math.max(0, x);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'tanh':
      return Math.tanh(x);
    case 'linear':
      return x;
    default:
      return x;
  }
}

/**
 * Apply activation function derivative
 */
export function applyActivationDerivative(x, activationType) {
  switch (activationType) {
    case 'relu':
      return x > 0 ? 1 : 0;
    case 'sigmoid':
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    case 'tanh':
      const t = Math.tanh(x);
      return 1 - t * t;
    case 'linear':
      return 1;
    default:
      return 1;
  }
}

/**
 * Forward pass through a single layer
 */
export function forwardPassLayer(input, weights, biases, activation) {
  const output = [];
  for (let i = 0; i < weights.length; i++) {
    let sum = biases[i] || 0;
    for (let j = 0; j < input.length; j++) {
      sum += weights[i][j] * input[j];
    }
    output.push(applyActivation(sum, activation));
  }
  return output;
}

/**
 * Forward pass through entire network
 */
export function forwardPassNetwork(input, network) {
  let current = input;
  const activations = [input];
  
  for (const layer of network) {
    current = forwardPassLayer(
      current,
      layer.weights,
      layer.biases,
      layer.activation
    );
    activations.push(current);
  }
  
  return { output: current, activations };
}

/**
 * Calculate mean squared error loss
 */
export function mseLoss(predictions, targets) {
  if (predictions.length !== targets.length) {
    throw new Error('Predictions and targets must have same length');
  }
  let sum = 0;
  for (let i = 0; i < predictions.length; i++) {
    sum += Math.pow(predictions[i] - targets[i], 2);
  }
  return sum / predictions.length;
}

/**
 * Calculate cross-entropy loss (for classification)
 */
export function crossEntropyLoss(predictions, targets) {
  if (predictions.length !== targets.length) {
    throw new Error('Predictions and targets must have same length');
  }
  let sum = 0;
  for (let i = 0; i < predictions.length; i++) {
    const p = Math.max(1e-15, Math.min(1 - 1e-15, predictions[i]));
    sum -= targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p);
  }
  return sum / predictions.length;
}

/**
 * Simple backpropagation (simplified for visualization)
 */
export function backpropagate(network, activations, targets, lossType = 'mse') {
  const gradients = [];
  const output = activations[activations.length - 1];
  
  // Output layer error
  let error = [];
  if (lossType === 'mse') {
    for (let i = 0; i < output.length; i++) {
      error.push(2 * (output[i] - targets[i]));
    }
  } else {
    // Cross-entropy derivative
    for (let i = 0; i < output.length; i++) {
      error.push(output[i] - targets[i]);
    }
  }
  
  // Backpropagate through layers
  for (let l = network.length - 1; l >= 0; l--) {
    const layer = network[l];
    const layerInput = activations[l];
    const layerOutput = activations[l + 1];
    
    const layerGradients = {
      weights: [],
      biases: []
    };
    
    for (let i = 0; i < layer.weights.length; i++) {
      const activationDeriv = applyActivationDerivative(
        layerOutput[i],
        layer.activation
      );
      const delta = error[i] * activationDeriv;
      
      layerGradients.biases.push(delta);
      layerGradients.weights.push([]);
      
      for (let j = 0; j < layer.weights[i].length; j++) {
        layerGradients.weights[i].push(delta * layerInput[j]);
      }
    }
    
    gradients.unshift(layerGradients);
    
    // Propagate error to previous layer
    const newError = [];
    for (let j = 0; j < layerInput.length; j++) {
      let sum = 0;
      for (let i = 0; i < layer.weights.length; i++) {
        const activationDeriv = applyActivationDerivative(
          layerOutput[i],
          layer.activation
        );
        sum += error[i] * activationDeriv * layer.weights[i][j];
      }
      newError.push(sum);
    }
    error = newError;
  }
  
  return gradients;
}

/**
 * Create a simple neural network structure
 */
export function createNetwork(layerSizes, activations) {
  const network = [];
  
  for (let i = 1; i < layerSizes.length; i++) {
    const inputSize = layerSizes[i - 1];
    const outputSize = layerSizes[i];
    
    // Initialize weights randomly
    const weights = [];
    for (let j = 0; j < outputSize; j++) {
      weights.push([]);
      for (let k = 0; k < inputSize; k++) {
        weights[j].push((Math.random() - 0.5) * 0.1);
      }
    }
    
    // Initialize biases to zero
    const biases = new Array(outputSize).fill(0);
    
    network.push({
      weights,
      biases,
      activation: activations[i - 1] || 'relu'
    });
  }
  
  return network;
}

/**
 * Update network weights using gradients (gradient descent step)
 */
export function updateWeights(network, gradients, learningRate) {
  for (let l = 0; l < network.length; l++) {
    const layer = network[l];
    const grad = gradients[l];
    
    for (let i = 0; i < layer.weights.length; i++) {
      layer.biases[i] -= learningRate * grad.biases[i];
      for (let j = 0; j < layer.weights[i].length; j++) {
        layer.weights[i][j] -= learningRate * grad.weights[i][j];
      }
    }
  }
}

/**
 * Calculate attention scores (simplified for visualization)
 */
export function calculateAttention(query, keys, values) {
  const scores = [];
  for (let i = 0; i < keys.length; i++) {
    // Dot product attention (simplified)
    let score = 0;
    for (let j = 0; j < query.length; j++) {
      score += query[j] * keys[i][j];
    }
    scores.push(score);
  }
  
  // Softmax
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp(s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const attentionWeights = expScores.map(e => e / sumExp);
  
  // Weighted sum of values
  const output = [];
  for (let j = 0; j < values[0].length; j++) {
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += attentionWeights[i] * values[i][j];
    }
    output.push(sum);
  }
  
  return { attentionWeights, output };
}

/**
 * Generate sample training data
 */
export function generateTrainingData(numSamples = 10) {
  const data = [];
  for (let i = 0; i < numSamples; i++) {
    const x = Math.random() * 2 - 1; // -1 to 1
    const y = Math.sin(x * Math.PI); // Simple function
    data.push({ input: [x], target: [y] });
  }
  return data;
}

