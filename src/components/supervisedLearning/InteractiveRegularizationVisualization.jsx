import React, { useRef, useEffect, useState } from 'react';
import * as sl from '../../utils/supervisedLearning';

export default function InteractiveRegularizationVisualization() {
  const canvasRef = useRef(null);
  const [regularizationType, setRegularizationType] = useState('l2'); // 'l1' or 'l2'
  const [lambda, setLambda] = useState(0.1);

  // Sample weights
  const weights = [2.5, -1.8, 3.2, -0.9, 1.5, -2.1, 0.8, -1.3];

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
    const padding = { top: 60, right: 40, bottom: 80, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Calculate regularization penalties
    const l1Penalty = sl.l1Regularization(weights, lambda);
    const l2Penalty = sl.l2Regularization(weights, lambda);

    // Draw weight bars
    const barWidth = (width - padding.left - padding.right) / weights.length;
    const maxWeight = Math.max(...weights.map(Math.abs));
    const barHeight = (height - padding.top - padding.bottom) / 2;

    weights.forEach((weight, i) => {
      const x = padding.left + i * barWidth;
      const barH = (Math.abs(weight) / maxWeight) * barHeight;
      const y = height - padding.bottom - barH;

      // Original weight (top)
      ctx.fillStyle = weight >= 0 ? '#3b82f6' : '#ef4444';
      ctx.fillRect(x + barWidth * 0.1, y, barWidth * 0.35, barH);

      // Regularized weight (bottom)
      let regularizedWeight;
      if (regularizationType === 'l1') {
        // L1: shrink towards zero
        regularizedWeight = weight > 0 
          ? Math.max(0, weight - lambda)
          : Math.min(0, weight + lambda);
      } else {
        // L2: shrink proportionally
        regularizedWeight = weight / (1 + lambda);
      }

      const regBarH = (Math.abs(regularizedWeight) / maxWeight) * barHeight;
      const regY = height - padding.bottom / 2 - regBarH;

      ctx.fillStyle = regularizedWeight >= 0 ? '#10b981' : '#f59e0b';
      ctx.fillRect(x + barWidth * 0.55, regY, barWidth * 0.35, regBarH);

      // Draw weight labels
      ctx.fillStyle = '#1f2937';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(weight.toFixed(1), x + barWidth / 2, y - 5);
      ctx.fillText(regularizedWeight.toFixed(1), x + barWidth / 2, regY - 5);
    });

    // Draw labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Original Weights', width / 2, padding.top - 20);
    ctx.fillText('Regularized Weights', width / 2, height - padding.bottom / 2 - 20);

    // Draw weight index labels
    ctx.font = '10px sans-serif';
    weights.forEach((_, i) => {
      const x = padding.left + i * barWidth + barWidth / 2;
      ctx.fillText(`w${i}`, x, height - padding.bottom + 15);
    });

    // Draw regularization info
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'left';
    const penalty = regularizationType === 'l1' ? l1Penalty : l2Penalty;
    ctx.fillText(`${regularizationType.toUpperCase()} Penalty: ${penalty.toFixed(3)}`, padding.left, padding.top + 20);
    
    const originalNorm = regularizationType === 'l1' 
      ? weights.reduce((sum, w) => sum + Math.abs(w), 0)
      : Math.sqrt(weights.reduce((sum, w) => sum + w * w, 0));
    const regularizedNorm = regularizationType === 'l1'
      ? weights.map(w => w > 0 ? Math.max(0, w - lambda) : Math.min(0, w + lambda)).reduce((sum, w) => sum + Math.abs(w), 0)
      : Math.sqrt(weights.map(w => w / (1 + lambda)).reduce((sum, w) => sum + w * w, 0));
    
    ctx.fillText(`Original ${regularizationType.toUpperCase()} Norm: ${originalNorm.toFixed(3)}`, padding.left, padding.top + 35);
    ctx.fillText(`Regularized ${regularizationType.toUpperCase()} Norm: ${regularizedNorm.toFixed(3)}`, padding.left, padding.top + 50);

    // Draw legend
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(width - padding.right - 100, padding.top + 20, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.font = '11px sans-serif';
    ctx.fillText('Positive Weight', width - padding.right - 80, padding.top + 32);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(width - padding.right - 100, padding.top + 40, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Negative Weight', width - padding.right - 80, padding.top + 52);

    ctx.fillStyle = '#10b981';
    ctx.fillRect(width - padding.right - 100, padding.top + 60, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Regularized +', width - padding.right - 80, padding.top + 72);

    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(width - padding.right - 100, padding.top + 80, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Regularized -', width - padding.right - 80, padding.top + 92);

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${regularizationType.toUpperCase()} Regularization (λ = ${lambda.toFixed(2)})`, width / 2, 20);
  }, [regularizationType, lambda]);

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Adjust regularization type and strength to see how weights are penalized!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Regularization Type
          </label>
          <select
            value={regularizationType}
            onChange={(e) => setRegularizationType(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value="l1">L1 Regularization (Lasso)</option>
            <option value="l2">L2 Regularization (Ridge)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Regularization Strength (λ): {lambda.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>0 (No regularization)</span>
            <span>0.5</span>
            <span>1.0 (Strong regularization)</span>
          </div>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Explanation */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Regularization Explained:</h4>
        <div className="space-y-3 text-sm text-gray-700">
          <div>
            <strong>L1 Regularization (Lasso):</strong>
            <ul className="list-disc list-inside ml-2 mt-1 space-y-1">
              <li>Penalty: λ × Σ|w| (sum of absolute values)</li>
              <li>Effect: Shrinks weights towards zero, can set weights to exactly zero (feature selection)</li>
              <li>Use case: When you want to remove irrelevant features</li>
            </ul>
          </div>
          <div>
            <strong>L2 Regularization (Ridge):</strong>
            <ul className="list-disc list-inside ml-2 mt-1 space-y-1">
              <li>Penalty: λ × Σw² (sum of squared values)</li>
              <li>Effect: Shrinks weights proportionally, keeps all features but reduces their impact</li>
              <li>Use case: When all features might be relevant but you want to prevent overfitting</li>
            </ul>
          </div>
          <div>
            <strong>Why Regularize?</strong> Prevents overfitting by penalizing large weights, encouraging simpler models that generalize better.
          </div>
        </div>
      </div>
    </div>
  );
}

