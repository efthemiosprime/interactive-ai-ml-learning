import React, { useRef, useEffect, useState } from 'react';
import * as sl from '../../utils/supervisedLearning';

export default function InteractiveModelEvaluationVisualization() {
  const canvasRef = useRef(null);
  const [threshold, setThreshold] = useState(0.5);
  const [viewMode, setViewMode] = useState('confusion'); // 'confusion' or 'roc'

  // Sample predictions and actuals
  const [predictions] = useState(() => 
    Array.from({ length: 100 }, () => Math.random())
  );
  const [actuals] = useState(() => 
    Array.from({ length: 100 }, () => Math.random() < 0.6 ? 1 : 0)
  );

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

    if (viewMode === 'confusion') {
      // Draw Confusion Matrix
      const cm = sl.confusionMatrix(predictions, actuals, threshold);
      const cellWidth = (width - padding.left - padding.right) / 2;
      const cellHeight = (height - padding.top - padding.bottom) / 2;
      
      const startX = padding.left;
      const startY = padding.top;

      // Draw grid
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 2;
      ctx.strokeRect(startX, startY, cellWidth * 2, cellHeight * 2);
      ctx.beginPath();
      ctx.moveTo(startX + cellWidth, startY);
      ctx.lineTo(startX + cellWidth, startY + cellHeight * 2);
      ctx.moveTo(startX, startY + cellHeight);
      ctx.lineTo(startX + cellWidth * 2, startY + cellHeight);
      ctx.stroke();

      // Draw cells with colors
      // True Negative (top-left)
      ctx.fillStyle = '#10b981';
      ctx.fillRect(startX, startY, cellWidth, cellHeight);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 24px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('TN', startX + cellWidth / 2, startY + cellHeight / 2 - 10);
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText(cm.tn.toString(), startX + cellWidth / 2, startY + cellHeight / 2 + 15);

      // False Positive (top-right)
      ctx.fillStyle = '#f59e0b';
      ctx.fillRect(startX + cellWidth, startY, cellWidth, cellHeight);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 24px sans-serif';
      ctx.fillText('FP', startX + cellWidth * 1.5, startY + cellHeight / 2 - 10);
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText(cm.fp.toString(), startX + cellWidth * 1.5, startY + cellHeight / 2 + 15);

      // False Negative (bottom-left)
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(startX, startY + cellHeight, cellWidth, cellHeight);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 24px sans-serif';
      ctx.fillText('FN', startX + cellWidth / 2, startY + cellHeight * 1.5 - 10);
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText(cm.fn.toString(), startX + cellWidth / 2, startY + cellHeight * 1.5 + 15);

      // True Positive (bottom-right)
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(startX + cellWidth, startY + cellHeight, cellWidth, cellHeight);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 24px sans-serif';
      ctx.fillText('TP', startX + cellWidth * 1.5, startY + cellHeight * 1.5 - 10);
      ctx.font = 'bold 18px sans-serif';
      ctx.fillText(cm.tp.toString(), startX + cellWidth * 1.5, startY + cellHeight * 1.5 + 15);

      // Draw labels
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Predicted: Negative', startX + cellWidth, startY - 20);
      ctx.fillText('Predicted: Positive', startX + cellWidth * 1.5, startY - 20);
      
      ctx.save();
      ctx.translate(startX - 30, startY + cellHeight);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Actual: Negative', 0, 0);
      ctx.restore();

      ctx.save();
      ctx.translate(startX - 30, startY + cellHeight * 1.5);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Actual: Positive', 0, 0);
      ctx.restore();

      // Draw metrics
      const accuracy = sl.accuracy(predictions, actuals, threshold);
      const precision = sl.precision(predictions, actuals, threshold);
      const recall = sl.recall(predictions, actuals, threshold);
      const f1 = sl.f1Score(predictions, actuals, threshold);

      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Accuracy: ${accuracy.toFixed(3)}`, startX, height - padding.bottom + 20);
      ctx.fillText(`Precision: ${precision.toFixed(3)}`, startX, height - padding.bottom + 35);
      ctx.fillText(`Recall: ${recall.toFixed(3)}`, startX + cellWidth, height - padding.bottom + 20);
      ctx.fillText(`F1-Score: ${f1.toFixed(3)}`, startX + cellWidth, height - padding.bottom + 35);

    } else {
      // Draw ROC Curve
      const rocPoints = sl.rocCurve(predictions, actuals, 100);
      const aucValue = sl.auc(rocPoints);

      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;

      const toScreenX = (fpr) => padding.left + fpr * plotWidth;
      const toScreenY = (tpr) => padding.top + plotHeight - tpr * plotHeight;

      // Draw diagonal line (random classifier)
      ctx.strokeStyle = '#9ca3af';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, padding.top);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw ROC curve
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      let firstPoint = true;
      rocPoints.forEach(point => {
        const screenX = toScreenX(point.fpr);
        const screenY = toScreenY(point.tpr);
        if (firstPoint) {
          ctx.moveTo(screenX, screenY);
          firstPoint = false;
        } else {
          ctx.lineTo(screenX, screenY);
        }
      });
      ctx.stroke();

      // Fill area under curve
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.beginPath();
      ctx.moveTo(padding.left, height - padding.bottom);
      rocPoints.forEach(point => {
        ctx.lineTo(toScreenX(point.fpr), toScreenY(point.tpr));
      });
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.closePath();
      ctx.fill();

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
      ctx.fillText('False Positive Rate (FPR)', width / 2, height - padding.bottom + 40);
      
      ctx.save();
      ctx.translate(padding.left - 30, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('True Positive Rate (TPR)', 0, 0);
      ctx.restore();

      // Draw AUC value
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`AUC: ${aucValue.toFixed(3)}`, padding.left, padding.top + 20);

      // Draw threshold point
      const thresholdPoint = rocPoints.find(p => Math.abs(p.threshold - threshold) < 0.01) || rocPoints[Math.round(threshold * 100)];
      if (thresholdPoint) {
        const screenX = toScreenX(thresholdPoint.fpr);
        const screenY = toScreenY(thresholdPoint.tpr);
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(screenX, screenY, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.fillStyle = '#1f2937';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Threshold: ${threshold.toFixed(2)}`, screenX + 10, screenY - 5);
        ctx.fillText(`TPR: ${thresholdPoint.tpr.toFixed(3)}, FPR: ${thresholdPoint.fpr.toFixed(3)}`, screenX + 10, screenY + 10);
      }
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(viewMode === 'confusion' ? 'Confusion Matrix' : 'ROC Curve', width / 2, 20);
  }, [threshold, viewMode, predictions, actuals]);

  return (
    <div className="space-y-4">
      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-green-800">
          💡 <strong>Interactive:</strong> Adjust the threshold to see how it affects model performance metrics!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-green-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            View Mode
          </label>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value="confusion">Confusion Matrix</option>
            <option value="roc">ROC Curve</option>
          </select>
        </div>

        {viewMode === 'confusion' && (
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Classification Threshold: {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>
        )}

        {viewMode === 'roc' && (
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Threshold (for point on curve): {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Metrics Info */}
      {viewMode === 'confusion' && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 mb-2">Metrics Explained:</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li><strong>Accuracy:</strong> (TP + TN) / Total - Overall correctness</li>
            <li><strong>Precision:</strong> TP / (TP + FP) - Of predicted positives, how many are correct?</li>
            <li><strong>Recall:</strong> TP / (TP + FN) - Of actual positives, how many did we catch?</li>
            <li><strong>F1-Score:</strong> 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean</li>
          </ul>
        </div>
      )}

      {viewMode === 'roc' && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 mb-2">ROC Curve Explained:</h4>
          <ul className="text-sm text-gray-700 space-y-1">
            <li><strong>TPR (True Positive Rate):</strong> Recall - How many positives we correctly identify</li>
            <li><strong>FPR (False Positive Rate):</strong> FP / (FP + TN) - How many negatives we incorrectly classify as positive</li>
            <li><strong>AUC (Area Under Curve):</strong> Measures overall classifier performance (1.0 = perfect, 0.5 = random)</li>
            <li><strong>Diagonal line:</strong> Represents a random classifier (AUC = 0.5)</li>
          </ul>
        </div>
      )}
    </div>
  );
}

