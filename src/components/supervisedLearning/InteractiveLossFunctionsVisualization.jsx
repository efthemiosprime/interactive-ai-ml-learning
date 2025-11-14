import React, { useRef, useEffect, useState } from 'react';
import * as sl from '../../utils/supervisedLearning';

export default function InteractiveLossFunctionsVisualization() {
  const canvasRef = useRef(null);
  const [lossType, setLossType] = useState('mse');
  const [hoveredX, setHoveredX] = useState(null);

  // Sample data
  const actual = 1.0;
  const predictions = Array.from({ length: 200 }, (_, i) => (i / 100) - 1); // -1 to 1

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

    // Calculate loss values
    const actuals = predictions.map(() => actual);
    let lossValues = [];
    let maxLoss = 0;

    if (lossType === 'mse') {
      lossValues = predictions.map(pred => sl.meanSquaredError([pred], [actual]));
    } else if (lossType === 'mae') {
      lossValues = predictions.map(pred => sl.meanAbsoluteError([pred], [actual]));
    } else if (lossType === 'cross-entropy') {
      // For binary cross-entropy, convert predictions to probabilities
      const probs = predictions.map(p => (p + 1) / 2); // Map -1 to 1 -> 0 to 1
      lossValues = probs.map(prob => sl.binaryCrossEntropy([prob], [1]));
    } else if (lossType === 'hinge') {
      // Hinge loss expects predictions in -1 to 1 range, actuals as -1 or 1
      lossValues = predictions.map(pred => sl.hingeLoss([pred], [1]));
    }

    maxLoss = Math.max(...lossValues);

    // Convert to screen coordinates
    const toScreenX = (x) => padding.left + ((x - predictions[0]) / (predictions[predictions.length - 1] - predictions[0])) * (width - padding.left - padding.right);
    const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - (y / maxLoss) * (height - padding.top - padding.bottom);

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
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * (height - padding.top - padding.bottom);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Draw loss curve
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    let firstPoint = true;
    predictions.forEach((pred, i) => {
      const screenX = toScreenX(pred);
      const screenY = toScreenY(lossValues[i]);
      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    });
    ctx.stroke();

    // Draw actual value line
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    const actualX = toScreenX(actual);
    ctx.beginPath();
    ctx.moveTo(actualX, padding.top);
    ctx.lineTo(actualX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw minimum point
    const minIndex = lossValues.indexOf(Math.min(...lossValues));
    const minX = toScreenX(predictions[minIndex]);
    const minY = toScreenY(lossValues[minIndex]);
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(minX, minY, 6, 0, 2 * Math.PI);
    ctx.fill();

    // Draw hover point
    if (hoveredX !== null) {
      const worldX = predictions[0] + ((hoveredX - padding.left) / (width - padding.left - padding.right)) * (predictions[predictions.length - 1] - predictions[0]);
      const index = Math.round((worldX - predictions[0]) / (predictions[1] - predictions[0]));
      if (index >= 0 && index < lossValues.length) {
        const hoverX = toScreenX(predictions[index]);
        const hoverY = toScreenY(lossValues[index]);
        
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(hoverX, hoverY, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Draw label
        ctx.fillStyle = '#1f2937';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Prediction: ${predictions[index].toFixed(2)}`, hoverX + 10, hoverY - 10);
        ctx.fillText(`Loss: ${lossValues[index].toFixed(3)}`, hoverX + 10, hoverY + 5);
      }
    }

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Prediction Value', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Loss Value', 0, 0);
    ctx.restore();

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const lossLabels = {
      mse: 'Mean Squared Error (MSE)',
      mae: 'Mean Absolute Error (MAE)',
      'cross-entropy': 'Binary Cross-Entropy',
      hinge: 'Hinge Loss'
    };
    ctx.fillText(lossLabels[lossType], width / 2, 20);

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.fillText('Actual Value', width - padding.right - 100, padding.top + 20);
    ctx.strokeStyle = '#10b981';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 30);
    ctx.lineTo(width - padding.right - 50, padding.top + 30);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [lossType, hoveredX]);

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const padding = 60;
    const x = e.clientX - rect.left;
    
    if (x >= padding && x <= rect.width - padding) {
      setHoveredX(x);
    } else {
      setHoveredX(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredX(null);
  };

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Hover over the curve to see loss values at different predictions!
        </p>
      </div>

      {/* Loss Type Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Loss Function Type
        </label>
        <select
          value={lossType}
          onChange={(e) => setLossType(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
        >
          <option value="mse">Mean Squared Error (MSE) - Regression</option>
          <option value="mae">Mean Absolute Error (MAE) - Regression</option>
          <option value="cross-entropy">Binary Cross-Entropy - Classification</option>
          <option value="hinge">Hinge Loss - SVM</option>
        </select>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg cursor-crosshair"
        style={{ height: '500px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />

      {/* Loss Function Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">About {lossType === 'mse' ? 'MSE' : lossType === 'mae' ? 'MAE' : lossType === 'cross-entropy' ? 'Cross-Entropy' : 'Hinge Loss'}:</h4>
        <p className="text-sm text-gray-700">
          {lossType === 'mse' && 'MSE = (1/n)Σ(y_pred - y_true)². Penalizes large errors more. Used for regression.'}
          {lossType === 'mae' && 'MAE = (1/n)Σ|y_pred - y_true|. Linear penalty. Robust to outliers.'}
          {lossType === 'cross-entropy' && 'Cross-Entropy = -Σ[y_true·log(y_pred) + (1-y_true)·log(1-y_pred)]. Used for binary classification.'}
          {lossType === 'hinge' && 'Hinge Loss = max(0, 1 - y_true·y_pred). Used for Support Vector Machines (SVM).'}
        </p>
      </div>
    </div>
  );
}

