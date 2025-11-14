import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function LinearRegressionVisualization() {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [slope, setSlope] = useState(1);
  const [intercept, setIntercept] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [learningRate, setLearningRate] = useState(0.01);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0);
  const [showFormula, setShowFormula] = useState(true);

  useEffect(() => {
    generateSampleData();
  }, []);

  useEffect(() => {
    drawGraph();
  }, [dataPoints, slope, intercept, loss]);

  const generateSampleData = () => {
    const points = [];
    const trueSlope = 2;
    const trueIntercept = 1;
    
    for (let i = 0; i < 20; i++) {
      const x = Math.random() * 8 - 2;
      const y = trueSlope * x + trueIntercept + (Math.random() - 0.5) * 2;
      points.push({ x, y });
    }
    
    setDataPoints(points);
    setSlope(0.5);
    setIntercept(0);
  };

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 60;

    ctx.clearRect(0, 0, width, height);

    // Find data bounds
    const xMin = Math.min(...dataPoints.map(p => p.x), -2);
    const xMax = Math.max(...dataPoints.map(p => p.x), 6);
    const yMin = Math.min(...dataPoints.map(p => p.y), -2);
    const yMax = Math.max(...dataPoints.map(p => p.y), 15);
    
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    
    // X-axis
    const yAxisY = height - padding;
    ctx.beginPath();
    ctx.moveTo(padding, yAxisY);
    ctx.lineTo(width - padding, yAxisY);
    ctx.stroke();

    // Y-axis
    const xAxisX = padding;
    ctx.beginPath();
    ctx.moveTo(xAxisX, padding);
    ctx.lineTo(xAxisX, height - padding);
    ctx.stroke();

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (width - 2 * padding);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i / 10) * (height - 2 * padding);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw regression line: y = slope * x + intercept
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    const x1 = xMin;
    const y1 = slope * x1 + intercept;
    const x2 = xMax;
    const y2 = slope * x2 + intercept;
    
    const canvasX1 = padding + ((x1 - xMin) / xRange) * (width - 2 * padding);
    const canvasY1 = yAxisY - ((y1 - yMin) / yRange) * (height - 2 * padding);
    const canvasX2 = padding + ((x2 - xMin) / xRange) * (width - 2 * padding);
    const canvasY2 = yAxisY - ((y2 - yMin) / yRange) * (height - 2 * padding);
    
    ctx.moveTo(canvasX1, canvasY1);
    ctx.lineTo(canvasX2, canvasY2);
    ctx.stroke();

    // Draw data points
    ctx.fillStyle = '#ef4444';
    dataPoints.forEach(point => {
      const canvasX = padding + ((point.x - xMin) / xRange) * (width - 2 * padding);
      const canvasY = yAxisY - ((point.y - yMin) / yRange) * (height - 2 * padding);
      
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
      ctx.fill();

      // Draw error line (distance to regression line)
      const predictedY = slope * point.x + intercept;
      const predictedCanvasY = yAxisY - ((predictedY - yMin) / yRange) * (height - 2 * padding);
      
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(canvasX, canvasY);
      ctx.lineTo(canvasX, predictedCanvasY);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Labels
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.fillText('x', width - padding + 10, yAxisY + 20);
    ctx.fillText('y', xAxisX - 30, padding - 10);
  };

  const calculateLoss = () => {
    // Mean Squared Error: MSE = (1/m) Σ(y_pred - y_true)²
    let totalLoss = 0;
    dataPoints.forEach(point => {
      const predicted = slope * point.x + intercept;
      totalLoss += Math.pow(predicted - point.y, 2);
    });
    return totalLoss / dataPoints.length;
  };

  const train = async () => {
    setIsTraining(true);
    setEpoch(0);

    for (let e = 0; e < 100; e++) {
      await new Promise(resolve => setTimeout(resolve, 50));

      // Calculate gradients using calculus
      // ∂L/∂slope = (2/m) Σ(y_pred - y_true) · x
      // ∂L/∂intercept = (2/m) Σ(y_pred - y_true)
      
      let gradientSlope = 0;
      let gradientIntercept = 0;
      const m = dataPoints.length;

      dataPoints.forEach(point => {
        const predicted = slope * point.x + intercept;
        const error = predicted - point.y;
        gradientSlope += error * point.x;
        gradientIntercept += error;
      });

      gradientSlope *= 2 / m;
      gradientIntercept *= 2 / m;

      // Update parameters using gradient descent
      // θ = θ - α · ∇L
      const newSlope = slope - learningRate * gradientSlope;
      const newIntercept = intercept - learningRate * gradientIntercept;

      setSlope(newSlope);
      setIntercept(newIntercept);
      setLoss(calculateLoss());
      setEpoch(e + 1);

      // Stop if converged
      if (Math.abs(gradientSlope) < 0.001 && Math.abs(gradientIntercept) < 0.001) {
        break;
      }
    }

    setIsTraining(false);
  };

  const codeExample = `# Linear Regression with Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.slope = 0
        self.intercept = 0
    
    def predict(self, X):
        """
        Linear Algebra: Matrix multiplication
        ŷ = X · θ
        
        where:
        - X = [1, x] (design matrix with bias column)
        - θ = [intercept, slope] (parameters)
        """
        return X * self.slope + self.intercept
    
    def loss(self, X, y):
        """
        Mean Squared Error (MSE)
        L = (1/m) Σ(ŷ - y)²
        
        where:
        - m = number of samples
        - ŷ = predicted values
        - y = true values
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def gradient(self, X, y):
        """
        Calculate gradients using Calculus
        
        ∂L/∂slope = (2/m) Σ(ŷ - y) · x
        ∂L/∂intercept = (2/m) Σ(ŷ - y)
        
        These are partial derivatives of the loss function
        """
        m = len(X)
        predictions = self.predict(X)
        errors = predictions - y
        
        gradient_slope = (2 / m) * np.sum(errors * X)
        gradient_intercept = (2 / m) * np.sum(errors)
        
        return gradient_slope, gradient_intercept
    
    def fit(self, X, y, epochs=100):
        """
        Train using Gradient Descent
        
        Formula: θ = θ - α · ∇L
        
        where:
        - θ = parameters [slope, intercept]
        - α = learning rate
        - ∇L = gradient vector
        """
        for epoch in range(epochs):
            # Calculate gradients
            grad_slope, grad_intercept = self.gradient(X, y)
            
            # Update parameters (Gradient Descent)
            self.slope -= self.learning_rate * grad_slope
            self.intercept -= self.learning_rate * grad_intercept
            
            # Calculate loss
            current_loss = self.loss(X, y)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.4f}")
            
            # Check convergence
            if abs(grad_slope) < 0.001 and abs(grad_intercept) < 0.001:
                break
        
        return self

# Example usage
# Generate sample data
np.random.seed(42)
X = np.random.randn(20) * 3
y = 2 * X + 1 + np.random.randn(20) * 0.5

# Train model
model = LinearRegression(learning_rate=0.01)
model.fit(X, y, epochs=100)

print(f"Slope: {model.slope:.4f}")
print(f"Intercept: {model.intercept:.4f}")

# Visualize
plt.scatter(X, y, color='red', label='Data')
plt.plot(X, model.predict(X), color='blue', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()`;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">Linear Regression Visualization</h2>
        <p className="text-purple-800 text-sm">
          See how Linear Algebra and Calculus work together to fit a line to data points using gradient descent.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Learning Rate (α): {learningRate.toFixed(3)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={train}
              disabled={isTraining}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isTraining ? 'Training...' : 'Train Model'}
            </button>
            <button
              onClick={generateSampleData}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
            >
              New Data
            </button>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showFormula}
              onChange={(e) => setShowFormula(e.target.checked)}
              className="w-4 h-4"
            />
            <label className="text-sm font-semibold text-gray-700">Show Formulas</label>
          </div>
        </div>

        {/* Current Status */}
        {(epoch > 0 || loss > 0) && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <strong>Epoch:</strong> {epoch}
              </div>
              <div>
                <strong>Slope:</strong> {slope.toFixed(4)}
              </div>
              <div>
                <strong>Intercept:</strong> {intercept.toFixed(4)}
              </div>
              <div>
                <strong>Loss (MSE):</strong> {loss.toFixed(4)}
              </div>
            </div>
          </div>
        )}

        {/* Formula Display */}
        {showFormula && (
          <div className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
            <div className="text-sm space-y-2">
              <div>
                <strong>Regression Line:</strong> y = {slope.toFixed(3)}x + {intercept.toFixed(3)}
              </div>
              <div>
                <strong>Loss Function:</strong> MSE = (1/m) Σ(y_pred - y_true)²
              </div>
              <div>
                <strong>Gradients:</strong>
                <ul className="ml-4 mt-1 list-disc list-inside">
                  <li>∂L/∂slope = (2/m) Σ(y_pred - y_true) · x</li>
                  <li>∂L/∂intercept = (2/m) Σ(y_pred - y_true)</li>
                </ul>
              </div>
              <div>
                <strong>Update Rule:</strong> θ = θ - α · ∇L
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Canvas Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Data & Regression Line</h3>
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
            <div className="w-4 h-4 bg-red-500 rounded-full"></div>
            <span>Data Points</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500"></div>
            <span>Regression Line</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-500" style={{ borderStyle: 'dashed' }}></div>
            <span>Error (Distance to Line)</span>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Linear Regression Implementation</h3>
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
        <h3 className="font-semibold text-blue-900 mb-2">Mathematical Foundations:</h3>
        <div className="space-y-2 text-blue-800 text-sm">
          <div>
            <strong>Linear Algebra:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Prediction: ŷ = X · θ (matrix multiplication)</li>
              <li>X = design matrix, θ = parameter vector [intercept, slope]</li>
            </ul>
          </div>
          <div>
            <strong>Calculus:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Partial derivatives: ∂L/∂slope, ∂L/∂intercept</li>
              <li>Gradient: ∇L = [∂L/∂slope, ∂L/∂intercept]</li>
              <li>Gradient descent: θ = θ - α · ∇L</li>
            </ul>
          </div>
          <div>
            <strong>Statistics:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Mean Squared Error (MSE) as loss function</li>
              <li>Minimizing error to find best fit line</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

