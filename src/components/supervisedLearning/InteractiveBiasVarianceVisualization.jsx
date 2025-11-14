import React, { useRef, useEffect, useState } from 'react';

export default function InteractiveBiasVarianceVisualization() {
  const canvasRef = useRef(null);
  const [modelComplexity, setModelComplexity] = useState(1); // 1 = underfit, 2 = good, 3 = overfit

  // Generate sample data
  const generateData = () => {
    const data = [];
    for (let i = 0; i < 20; i++) {
      const x = (i / 20) * 4 - 2; // -2 to 2
      const trueY = 0.3 * x * x + 0.2 * x; // True function: quadratic
      const noisyY = trueY + (Math.random() - 0.5) * 0.5; // Add noise
      data.push({ x, y: noisyY, trueY });
    }
    return data;
  };

  const [data] = useState(generateData());

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

    const xMin = -2;
    const xMax = 2;
    const yMin = -1;
    const yMax = 3;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
    const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = padding.left + (i / 5) * (width - padding.left - padding.right);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }

    // Draw true function
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    let firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.1) {
      const trueY = 0.3 * x * x + 0.2 * x;
      const screenX = toScreenX(x);
      const screenY = toScreenY(trueY);
      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw model predictions based on complexity
    ctx.strokeStyle = modelComplexity === 1 ? '#ef4444' : modelComplexity === 2 ? '#3b82f6' : '#f59e0b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.1) {
      let predY;
      if (modelComplexity === 1) {
        // Underfit: linear model (high bias, low variance)
        predY = 0.1 * x + 0.5; // Simple linear
      } else if (modelComplexity === 2) {
        // Good fit: quadratic (balanced)
        predY = 0.3 * x * x + 0.2 * x;
      } else {
        // Overfit: high-degree polynomial (low bias, high variance)
        predY = 0.3 * x * x + 0.2 * x + 0.1 * Math.sin(x * 5) * Math.exp(-x * x);
      }
      const screenX = toScreenX(x);
      const screenY = toScreenY(predY);
      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();

    // Draw data points
    data.forEach(point => {
      ctx.fillStyle = '#6b7280';
      ctx.beginPath();
      ctx.arc(toScreenX(point.x), toScreenY(point.y), 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('X', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Y', 0, 0);
    ctx.restore();

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    ctx.strokeStyle = '#10b981';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 20);
    ctx.lineTo(width - padding.right - 50, padding.top + 20);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#10b981';
    ctx.fillText('True Function', width - padding.right - 45, padding.top + 25);

    ctx.strokeStyle = modelComplexity === 1 ? '#ef4444' : modelComplexity === 2 ? '#3b82f6' : '#f59e0b';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 40);
    ctx.lineTo(width - padding.right - 50, padding.top + 40);
    ctx.stroke();
    ctx.fillStyle = modelComplexity === 1 ? '#ef4444' : modelComplexity === 2 ? '#3b82f6' : '#f59e0b';
    ctx.fillText('Model Prediction', width - padding.right - 45, padding.top + 45);

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Bias-Variance Tradeoff', width / 2, 20);
  }, [modelComplexity, data]);

  const complexityLabels = {
    1: 'Underfitting (High Bias, Low Variance)',
    2: 'Good Fit (Balanced)',
    3: 'Overfitting (Low Bias, High Variance)'
  };

  const complexityDescriptions = {
    1: 'Model is too simple. It misses patterns in the data (high bias). Consistent predictions (low variance).',
    2: 'Model captures the true pattern well. Good balance between bias and variance.',
    3: 'Model is too complex. It fits noise in training data (low bias). Predictions vary a lot (high variance).'
  };

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Adjust model complexity to see the bias-variance tradeoff!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Model Complexity
          </label>
          <select
            value={modelComplexity}
            onChange={(e) => setModelComplexity(Number(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value={1}>Underfitting (High Bias, Low Variance)</option>
            <option value={2}>Good Fit (Balanced)</option>
            <option value={3}>Overfitting (Low Bias, High Variance)</option>
          </select>
        </div>

        <div className="bg-gray-50 rounded-lg p-3">
          <h4 className="font-semibold text-gray-900 mb-2">{complexityLabels[modelComplexity]}</h4>
          <p className="text-sm text-gray-700">{complexityDescriptions[modelComplexity]}</p>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Explanation */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Understanding Bias-Variance Tradeoff:</h4>
        <ul className="text-sm text-gray-700 space-y-2">
          <li><strong>Bias:</strong> Error from oversimplifying assumptions. High bias = model misses relevant patterns.</li>
          <li><strong>Variance:</strong> Error from sensitivity to small fluctuations. High variance = model fits noise.</li>
          <li><strong>Tradeoff:</strong> As model complexity increases, bias decreases but variance increases. Goal is to find the sweet spot.</li>
          <li><strong>Solution:</strong> Use validation set, cross-validation, and regularization to find optimal complexity.</li>
        </ul>
      </div>
    </div>
  );
}

