import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function GradientDescentVisualization() {
  const canvasRef = useRef(null);
  const [learningRate, setLearningRate] = useState(0.1);
  const [functionType, setFunctionType] = useState('quadratic');
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentPoint, setCurrentPoint] = useState({ x: -2, y: 0 });
  const [history, setHistory] = useState([]);
  const [iteration, setIteration] = useState(0);

  // Function definitions
  const functions = {
    quadratic: {
      fn: (x) => x * x + 2 * x + 1,
      derivative: (x) => 2 * x + 2,
      min: -1,
      range: { min: -4, max: 2 }
    },
    cubic: {
      fn: (x) => x * x * x - 3 * x,
      derivative: (x) => 3 * x * x - 3,
      min: 1,
      range: { min: -2.5, max: 2.5 }
    },
    complex: {
      fn: (x) => Math.sin(x) * x + 0.5 * x * x,
      derivative: (x) => Math.cos(x) * x + Math.sin(x) + x,
      min: -0.5,
      range: { min: -3, max: 3 }
    }
  };

  useEffect(() => {
    drawGraph();
  }, [functionType, currentPoint, history, learningRate]);

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 60;

    ctx.clearRect(0, 0, width, height);

    const func = functions[functionType];
    const { min: xMin, max: xMax } = func.range;
    const xRange = xMax - xMin;

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

    // Draw function
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    const step = (width - 2 * padding) / 200;
    let firstPoint = true;

    for (let i = 0; i <= 200; i++) {
      const x = xMin + (i / 200) * xRange;
      const y = func.fn(x);
      
      // Find y range
      let yMin = Infinity, yMax = -Infinity;
      for (let j = 0; j <= 200; j++) {
        const testX = xMin + (j / 200) * xRange;
        const testY = func.fn(testX);
        yMin = Math.min(yMin, testY);
        yMax = Math.max(yMax, testY);
      }
      const yRange = yMax - yMin || 1;

      const canvasX = padding + (i / 200) * (width - 2 * padding);
      const canvasY = yAxisY - ((y - yMin) / yRange) * (height - 2 * padding);

      if (firstPoint) {
        ctx.moveTo(canvasX, canvasY);
        firstPoint = false;
      } else {
        ctx.lineTo(canvasX, canvasY);
      }
    }
    ctx.stroke();

    // Draw minimum point
    const minY = func.fn(func.min);
    let yMin = Infinity, yMax = -Infinity;
    for (let j = 0; j <= 200; j++) {
      const testX = xMin + (j / 200) * xRange;
      const testY = func.fn(testX);
      yMin = Math.min(yMin, testY);
      yMax = Math.max(yMax, testY);
    }
    const yRange = yMax - yMin || 1;
    const minCanvasX = padding + ((func.min - xMin) / xRange) * (width - 2 * padding);
    const minCanvasY = yAxisY - ((minY - yMin) / yRange) * (height - 2 * padding);
    
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(minCanvasX, minCanvasY, 6, 0, 2 * Math.PI);
    ctx.fill();

    // Draw history path
    if (history.length > 1) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < history.length; i++) {
        const point = history[i];
        const canvasX = padding + ((point.x - xMin) / xRange) * (width - 2 * padding);
        const pointY = func.fn(point.x);
        const canvasY = yAxisY - ((pointY - yMin) / yRange) * (height - 2 * padding);
        
        if (i === 0) {
          ctx.moveTo(canvasX, canvasY);
        } else {
          ctx.lineTo(canvasX, canvasY);
        }
      }
      ctx.stroke();

      // Draw history points
      ctx.fillStyle = '#ef4444';
      for (let i = 0; i < history.length; i++) {
        const point = history[i];
        const canvasX = padding + ((point.x - xMin) / xRange) * (width - 2 * padding);
        const pointY = func.fn(point.x);
        const canvasY = yAxisY - ((pointY - yMin) / yRange) * (height - 2 * padding);
        
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Draw current point
    const currentCanvasX = padding + ((currentPoint.x - xMin) / xRange) * (width - 2 * padding);
    const currentY = func.fn(currentPoint.x);
    const currentCanvasY = yAxisY - ((currentY - yMin) / yRange) * (height - 2 * padding);

    // Draw gradient arrow
    const gradient = func.derivative(currentPoint.x);
    const arrowLength = 40;
    const arrowX = currentCanvasX + (gradient > 0 ? -arrowLength : arrowLength);
    const arrowY = currentCanvasY - Math.abs(gradient) * 10;

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(currentCanvasX, currentCanvasY);
    ctx.lineTo(arrowX, arrowY);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(arrowY - currentCanvasY, arrowX - currentCanvasX);
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle - Math.PI / 6), arrowY - 8 * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle + Math.PI / 6), arrowY - 8 * Math.sin(angle + Math.PI / 6));
    ctx.stroke();

    // Draw current point
    ctx.fillStyle = '#8b5cf6';
    ctx.beginPath();
    ctx.arc(currentCanvasX, currentCanvasY, 8, 0, 2 * Math.PI);
    ctx.fill();

    // Labels
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.fillText('x', width - padding + 10, yAxisY + 20);
    ctx.fillText('f(x)', xAxisX - 30, padding - 10);
    ctx.fillText(`Minimum at x = ${func.min.toFixed(2)}`, minCanvasX - 40, minCanvasY - 15);
  };

  const runGradientDescent = async () => {
    setIsAnimating(true);
    setHistory([]);
    setIteration(0);
    
    const func = functions[functionType];
    let x = currentPoint.x;
    const newHistory = [{ x, y: func.fn(x) }];

    for (let i = 0; i < 50; i++) {
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Gradient descent step: x = x - α * f'(x)
      const gradient = func.derivative(x);
      x = x - learningRate * gradient;
      
      newHistory.push({ x, y: func.fn(x) });
      setCurrentPoint({ x, y: func.fn(x) });
      setHistory([...newHistory]);
      setIteration(i + 1);

      // Stop if converged
      if (Math.abs(gradient) < 0.01) {
        break;
      }
    }

    setIsAnimating(false);
  };

  const reset = () => {
    setIsAnimating(false);
    setCurrentPoint({ x: -2, y: 0 });
    setHistory([]);
    setIteration(0);
  };

  const codeExample = `# Gradient Descent Algorithm

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, f_prime, x0, learning_rate=0.1, max_iterations=100):
    """
    Gradient Descent Algorithm
    
    Formula:
    x_{n+1} = x_n - α · f'(x_n)
    
    where:
    - x_n = current point
    - α = learning rate
    - f'(x_n) = derivative (gradient) at x_n
    """
    x = x0
    history = [x]
    
    for i in range(max_iterations):
        # Calculate gradient (derivative)
        gradient = f_prime(x)
        
        # Update: move in direction opposite to gradient
        x = x - learning_rate * gradient
        
        history.append(x)
        
        # Check convergence
        if abs(gradient) < 0.001:
            break
    
    return x, history

# Example: Minimize f(x) = x² + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def f_prime(x):
    return 2*x + 2  # Derivative: 2x + 2

# Run gradient descent
x_min, history = gradient_descent(f, f_prime, x0=-2, learning_rate=0.1)

print(f"Minimum found at x = {x_min:.4f}")
print(f"f(x_min) = {f(x_min):.4f}")
print(f"Iterations: {len(history)}")

# Visualize
x_vals = np.linspace(-4, 2, 100)
y_vals = [f(x) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x² + 2x + 1', linewidth=2)
plt.plot(history, [f(x) for x in history], 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.title('Gradient Descent Visualization')
plt.show()`;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">Gradient Descent Visualization</h2>
        <p className="text-purple-800 text-sm">
          Watch how gradient descent finds the minimum of a function step by step. See how learning rate affects convergence.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Function Type
            </label>
            <select
              value={functionType}
              onChange={(e) => {
                setFunctionType(e.target.value);
                reset();
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="quadratic">Quadratic: f(x) = x² + 2x + 1</option>
              <option value="cubic">Cubic: f(x) = x³ - 3x</option>
              <option value="complex">Complex: f(x) = sin(x)·x + 0.5x²</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Learning Rate (α): {learningRate.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="text-xs text-gray-500 mt-1">
              Small: Slow but stable | Large: Fast but may overshoot
            </div>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={runGradientDescent}
              disabled={isAnimating}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isAnimating ? 'Running...' : 'Run Gradient Descent'}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Current Status */}
        {iteration > 0 && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <strong>Iteration:</strong> {iteration}
              </div>
              <div>
                <strong>Current x:</strong> {currentPoint.x.toFixed(4)}
              </div>
              <div>
                <strong>f(x):</strong> {currentPoint.y.toFixed(4)}
              </div>
              <div>
                <strong>Gradient:</strong> {functions[functionType].derivative(currentPoint.x).toFixed(4)}
              </div>
              <div>
                <strong>Learning Rate:</strong> {learningRate.toFixed(2)}
              </div>
              <div>
                <strong>Step Size:</strong> {(learningRate * Math.abs(functions[functionType].derivative(currentPoint.x))).toFixed(4)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Canvas Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Function & Gradient Descent Path</h3>
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
            <div className="w-4 h-4 bg-blue-500 rounded"></div>
            <span>Function f(x)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
            <span>True Minimum</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
            <span>Current Point</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Path Taken</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-500"></div>
            <span>Gradient Direction</span>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Gradient Descent Implementation</h3>
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
        <h3 className="font-semibold text-blue-900 mb-2">Mathematical Formula:</h3>
        <div className="space-y-2 text-blue-800 text-sm">
          <div>
            <strong>Gradient Descent Update Rule:</strong>
            <p className="ml-4 mt-1">x<sub>{'{'}n+1{'}'}</sub> = x<sub>n</sub> - α · f'(x<sub>n</sub>)</p>
          </div>
          <div>
            <strong>Where:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>x<sub>n</sub> = current parameter value</li>
              <li>α (alpha) = learning rate</li>
              <li>f'(x<sub>n</sub>) = derivative (gradient) at x<sub>n</sub></li>
              <li>Negative sign = move in direction of steepest descent</li>
            </ul>
          </div>
          <div>
            <strong>Key Concepts:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Gradient points in direction of steepest ascent</li>
              <li>We move opposite to gradient (steepest descent)</li>
              <li>Learning rate controls step size</li>
              <li>Too small: slow convergence</li>
              <li>Too large: may overshoot minimum or diverge</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

