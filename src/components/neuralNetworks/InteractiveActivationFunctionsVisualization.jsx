import React, { useRef, useEffect, useState } from 'react';
import * as nn from '../../utils/neuralNetworks';

export default function InteractiveActivationFunctionsVisualization() {
  const canvasRef = useRef(null);
  const [selectedFunction, setSelectedFunction] = useState('relu');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 800;
    const displayHeight = rect.height > 0 ? rect.height : 500;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const xMin = -3;
    const xMax = 3;
    const yMin = -1.5;
    const yMax = 1.5;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
    const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      const x = toScreenX(i);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    for (let i = -1; i <= 1; i++) {
      const y = toScreenY(i);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    const zeroY = toScreenY(0);
    const zeroX = toScreenX(0);
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(width - padding.right, zeroY);
    ctx.moveTo(zeroX, padding.top);
    ctx.lineTo(zeroX, height - padding.bottom);
    ctx.stroke();

    // Draw function
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    const step = 0.1;
    let firstPoint = true;
    for (let x = xMin; x <= xMax; x += step) {
      const y = nn.applyActivation(x, selectedFunction);
      const screenX = toScreenX(x);
      const screenY = toScreenY(y);

      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();

    // Draw derivative
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    firstPoint = true;
    for (let x = xMin; x <= xMax; x += step) {
      const y = nn.applyActivationDerivative(x, selectedFunction);
      const screenX = toScreenX(x);
      const screenY = toScreenY(y);

      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const functionNames = {
      relu: 'ReLU (Rectified Linear Unit)',
      sigmoid: 'Sigmoid',
      tanh: 'Hyperbolic Tangent (Tanh)',
      linear: 'Linear'
    };
    ctx.fillText(functionNames[selectedFunction] || 'Activation Function', width / 2, 20);

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#8b5cf6';
    ctx.fillRect(width - padding.right - 100, padding.top + 20, 15, 3);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Function', width - padding.right - 80, padding.top + 25);
    
    ctx.fillStyle = '#ef4444';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 35);
    ctx.lineTo(width - padding.right - 85, padding.top + 35);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Derivative', width - padding.right - 80, padding.top + 38);

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('y', 0, 0);
    ctx.restore();
  }, [selectedFunction]);

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> Compare different activation functions and their derivatives!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Activation Function
        </label>
        <select
          value={selectedFunction}
          onChange={(e) => setSelectedFunction(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500"
        >
          <option value="relu">ReLU</option>
          <option value="sigmoid">Sigmoid</option>
          <option value="tanh">Tanh</option>
          <option value="linear">Linear</option>
        </select>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Activation Functions:</h4>
        <ul className="text-sm text-gray-700 space-y-2">
          <li><strong>ReLU:</strong> f(x) = max(0, x). Most common, solves vanishing gradient problem</li>
          <li><strong>Sigmoid:</strong> f(x) = 1/(1+e^(-x)). Outputs 0-1, used in binary classification</li>
          <li><strong>Tanh:</strong> f(x) = tanh(x). Outputs -1 to 1, zero-centered</li>
          <li><strong>Linear:</strong> f(x) = x. No transformation, rarely used in hidden layers</li>
        </ul>
      </div>
    </div>
  );
}

