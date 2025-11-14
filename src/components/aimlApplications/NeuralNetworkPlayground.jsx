import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function NeuralNetworkPlayground() {
  const canvasRef = useRef(null);
  const [networkConfig, setNetworkConfig] = useState({ layers: [2, 3, 1] });
  const [weights, setWeights] = useState([]);
  const [biases, setBiases] = useState([]);
  const [activations, setActivations] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [input, setInput] = useState([0.5, 0.3]);
  const [target, setTarget] = useState(0.8);
  const [showForward, setShowForward] = useState(true);
  const [showBackward, setShowBackward] = useState(false);

  useEffect(() => {
    initializeNetwork();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networkConfig]);

  useEffect(() => {
    if (weights.length > 0 && activations.length > 0) {
      drawNetwork();
    }
  }, [weights, biases, activations, showForward, showBackward, input, loss, networkConfig]);

  const sigmoid = (x) => {
    return 1 / (1 + Math.exp(-x));
  };

  const sigmoidDerivative = (x) => {
    const s = sigmoid(x);
    return s * (1 - s);
  };

  const forwardPass = (inputData, useWeights = weights, useBiases = biases) => {
    const { layers } = networkConfig;
    if (!useWeights || useWeights.length === 0) return [inputData];
    
    const activations = [inputData];
    
    for (let i = 0; i < useWeights.length; i++) {
      const layerActivations = [];
      for (let j = 0; j < useWeights[i].length; j++) {
        let sum = useBiases[i][j];
        for (let k = 0; k < activations[i].length; k++) {
          sum += activations[i][k] * useWeights[i][j][k];
        }
        layerActivations.push(sigmoid(sum));
      }
      activations.push(layerActivations);
    }
    
    return activations;
  };

  const initializeNetwork = () => {
    const { layers } = networkConfig;
    const newWeights = [];
    const newBiases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      const layerWeights = [];
      const layerBiases = [];
      
      for (let j = 0; j < layers[i + 1]; j++) {
        const neuronWeights = [];
        for (let k = 0; k < layers[i]; k++) {
          neuronWeights.push((Math.random() * 2 - 1) * 0.5);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push((Math.random() * 2 - 1) * 0.5);
      }
      
      newWeights.push(layerWeights);
      newBiases.push(layerBiases);
    }

    setWeights(newWeights);
    setBiases(newBiases);
    
    // Perform initial forward pass
    const acts = forwardPass(input, newWeights, newBiases);
    setActivations(acts);
  };

  const calculateLoss = (predicted, target) => {
    return 0.5 * Math.pow(predicted - target, 2);
  };

  const backwardPass = (activations, target) => {
    const { layers } = networkConfig;
    const gradients = [];
    
    // Output layer gradient
    const output = activations[activations.length - 1][0];
    const outputError = output - target;
    const outputGradient = outputError * sigmoidDerivative(
      activations[activations.length - 1][0]
    );
    gradients.push([outputGradient]);
    
    // Backpropagate through hidden layers
    for (let i = weights.length - 2; i >= 0; i--) {
      const layerGradients = [];
      for (let j = 0; j < layers[i + 1]; j++) {
        let gradient = 0;
        for (let k = 0; k < gradients[gradients.length - 1].length; k++) {
          gradient += gradients[gradients.length - 1][k] * weights[i + 1][k][j];
        }
        const activation = activations[i + 1][j];
        gradient *= sigmoidDerivative(activation);
        layerGradients.push(gradient);
      }
      gradients.push(layerGradients);
    }
    
    return gradients.reverse();
  };

  const trainStep = async () => {
    setIsTraining(true);
    const learningRate = 0.5;
    
    for (let e = 0; e < 10; e++) {
      // Forward pass
      const currentActivations = forwardPass(input);
      setActivations(currentActivations);
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Calculate loss
      const predicted = currentActivations[currentActivations.length - 1][0];
      const currentLoss = calculateLoss(predicted, target);
      setLoss(currentLoss);
      
      // Backward pass
      setShowBackward(true);
      const gradients = backwardPass(currentActivations, target);
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Update weights and biases
      const newWeights = [...weights];
      const newBiases = [...biases];
      
      for (let i = 0; i < weights.length; i++) {
        for (let j = 0; j < weights[i].length; j++) {
          // Update bias
          newBiases[i][j] -= learningRate * gradients[i][j];
          
          // Update weights
          for (let k = 0; k < weights[i][j].length; k++) {
            newWeights[i][j][k] -= learningRate * gradients[i][j] * currentActivations[i][k];
          }
        }
      }
      
      setWeights(newWeights);
      setBiases(newBiases);
      setEpoch(e + 1);
      await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    setIsTraining(false);
    setShowBackward(false);
  };

  const drawNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas || weights.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);

    const { layers } = networkConfig;
    if (!layers || layers.length === 0) return;
    
    const maxLayerSize = Math.max(...layers);
    const nodeRadius = 20;
    const layerSpacing = width / (layers.length + 1);
    const nodeSpacing = height / (maxLayerSize + 1);

    // Draw connections
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < layers.length - 1; i++) {
      const x1 = (i + 1) * layerSpacing;
      const x2 = (i + 2) * layerSpacing;
      
      for (let j = 0; j < layers[i]; j++) {
        const y1 = (j + 1) * nodeSpacing + (maxLayerSize - layers[i]) * nodeSpacing / 2;
        
        for (let k = 0; k < layers[i + 1]; k++) {
          const y2 = (k + 1) * nodeSpacing + (maxLayerSize - layers[i + 1]) * nodeSpacing / 2;
          
          // Color by weight
          const weight = weights[i]?.[k]?.[j] || 0;
          if (weight > 0) {
            ctx.strokeStyle = `rgba(34, 197, 94, ${Math.min(Math.abs(weight), 1)})`;
          } else {
            ctx.strokeStyle = `rgba(239, 68, 68, ${Math.min(Math.abs(weight), 1)})`;
          }
          
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (let i = 0; i < layers.length; i++) {
      const x = (i + 1) * layerSpacing;
      const layerSize = layers[i];
      const startY = (maxLayerSize - layerSize) * nodeSpacing / 2;
      
      for (let j = 0; j < layerSize; j++) {
        const y = startY + (j + 1) * nodeSpacing;
        
        // Node color based on activation
        const activation = activations[i]?.[j] || 0;
        const intensity = Math.min(Math.max(activation, 0), 1);
        ctx.fillStyle = `rgba(59, 130, 246, ${intensity})`;
        
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Node border
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Activation value
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(activation.toFixed(2), x, y + 4);
      }
    }

    // Layer labels
    ctx.fillStyle = '#000';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    const layerNames = ['Input', 'Hidden', 'Output'];
    for (let i = 0; i < layers.length; i++) {
      const x = (i + 1) * layerSpacing;
      const label = i === 0 ? 'Input' : (i === layers.length - 1 ? 'Output' : `Hidden ${i}`);
      ctx.fillText(label, x, height - 20);
    }
  };

  const codeExample = `# Neural Network Forward & Backward Pass

import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, layers):
        """
        Initialize neural network
        
        Layers: [input_size, hidden_size, ..., output_size]
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # Linear Algebra: Weight matrix W ∈ R^(output × input)
            w = np.random.randn(layers[i + 1], layers[i]) * 0.5
            b = np.random.randn(layers[i + 1]) * 0.5
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, x):
        """
        Forward Pass
        
        Formula for each layer:
        z = W · a + b  (Linear Algebra: matrix multiplication)
        a = σ(z)       (Activation function)
        
        where:
        - W = weight matrix
        - a = activations from previous layer
        - b = bias vector
        - σ = sigmoid activation
        """
        activations = [x]
        
        for i in range(len(self.weights)):
            # Linear transformation: z = W·a + b
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            # Activation: a = σ(z)
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations
    
    def backward(self, activations, target, learning_rate=0.5):
        """
        Backward Pass (Backpropagation)
        
        Uses Chain Rule from Calculus:
        ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
        
        Steps:
        1. Calculate output error: δ_output = (predicted - target)
        2. Backpropagate error through layers
        3. Update weights: w = w - α · ∂L/∂w
        """
        # Output layer gradient
        output = activations[-1]
        output_error = output - target
        gradients = [output_error * self.sigmoid_derivative(output)]
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Chain rule: gradient = next_gradient · weights · activation_derivative
            gradient = np.dot(self.weights[i + 1].T, gradients[0])
            gradient *= self.sigmoid_derivative(activations[i + 1])
            gradients.insert(0, gradient)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            # Gradient descent: w = w - α · gradient
            self.weights[i] -= learning_rate * np.outer(
                gradients[i], activations[i]
            )
            self.biases[i] -= learning_rate * gradients[i]
    
    def train(self, x, y, epochs=100):
        """Train the network"""
        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(x)
            
            # Calculate loss
            loss = 0.5 * np.sum((activations[-1] - y) ** 2)
            
            # Backward pass
            self.backward(activations, y)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return activations[-1]

# Example usage
network = SimpleNeuralNetwork([2, 3, 1])

# Training data
x = np.array([0.5, 0.3])
y = np.array([0.8])

# Train
prediction = network.train(x, y, epochs=100)
print(f"Final prediction: {prediction}")`;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">Neural Network Playground</h2>
        <p className="text-purple-800 text-sm">
          Interactive visualization of a neural network. Watch forward pass, backpropagation, and weight updates in real-time.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Network Architecture
            </label>
            <select
              value={JSON.stringify(networkConfig.layers)}
              onChange={(e) => {
                const layers = JSON.parse(e.target.value);
                setNetworkConfig({ layers });
                setEpoch(0);
                setLoss(0);
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="[2,3,1]">2 → 3 → 1 (Simple)</option>
              <option value="[2,4,3,1]">2 → 4 → 3 → 1 (Medium)</option>
              <option value="[3,5,4,2]">3 → 5 → 4 → 2 (Complex)</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Input 1
              </label>
              <input
                type="number"
                step="0.1"
                value={input[0]}
                onChange={(e) => setInput([parseFloat(e.target.value), input[1]])}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Input 2
              </label>
              <input
                type="number"
                step="0.1"
                value={input[1]}
                onChange={(e) => setInput([input[0], parseFloat(e.target.value)])}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Target Output
            </label>
            <input
              type="number"
              step="0.1"
              value={target}
              onChange={(e) => setTarget(parseFloat(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={() => {
                const acts = forwardPass(input);
                setActivations(acts);
                setShowForward(true);
                const pred = acts[acts.length - 1][0];
                setLoss(calculateLoss(pred, target));
              }}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
            >
              Forward Pass
            </button>
            <button
              onClick={trainStep}
              disabled={isTraining}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isTraining ? 'Training...' : 'Train (10 epochs)'}
            </button>
            <button
              onClick={initializeNetwork}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Status */}
        {(epoch > 0 || loss > 0) && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <strong>Epoch:</strong> {epoch}
              </div>
              <div>
                <strong>Loss:</strong> {loss.toFixed(4)}
              </div>
              <div>
                <strong>Prediction:</strong> {activations[activations.length - 1]?.[0]?.toFixed(4) || '0.0000'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Network Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Neural Network Visualization</h3>
        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={800}
            height={500}
            className="border-2 border-gray-300 rounded-lg"
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-blue-700"></div>
            <span>Neuron (activation shown)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500"></div>
            <span>Positive Weight</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500"></div>
            <span>Negative Weight</span>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Neural Network Implementation</h3>
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            customStyle={{ margin: 0, borderRadius: '0.5rem' }}
            showLineNumbers
          >
            {codeExample}
          </SyntaxHighlighter>
        </div>
      </div>

      {/* Mathematical Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Mathematical Formulas:</h3>
        <div className="space-y-2 text-blue-800 text-sm">
          <div>
            <strong>Forward Pass:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>z = W · a + b (Linear Algebra: matrix multiplication)</li>
              <li>a = σ(z) (Activation: sigmoid function)</li>
            </ul>
          </div>
          <div>
            <strong>Backward Pass (Chain Rule):</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w</li>
              <li>Weight update: w = w - α · ∂L/∂w</li>
              <li>Bias update: b = b - α · ∂L/∂b</li>
            </ul>
          </div>
          <div>
            <strong>Key Concepts:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Linear Algebra: Matrix operations for forward pass</li>
              <li>Calculus: Chain rule for backpropagation</li>
              <li>Gradient Descent: Update weights to minimize loss</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

