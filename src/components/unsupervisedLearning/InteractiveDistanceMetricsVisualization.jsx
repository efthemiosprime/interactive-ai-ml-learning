import React, { useRef, useEffect, useState } from 'react';
import * as ul from '../../utils/unsupervisedLearning';

export default function InteractiveDistanceMetricsVisualization() {
  const canvasRef = useRef(null);
  const [metric, setMetric] = useState('euclidean');
  const [point1, setPoint1] = useState([-2, -1]);
  const [point2, setPoint2] = useState([2, 2]);

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

    const xMin = -5;
    const xMax = 5;
    const yMin = -5;
    const yMax = 5;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
    const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i / 10) * (width - padding.left - padding.right);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    for (let i = 0; i <= 10; i++) {
      const y = padding.top + (i / 10) * (height - padding.top - padding.bottom);
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

    // Calculate distance
    let distance = 0;
    if (metric === 'euclidean') {
      distance = ul.euclideanDistance(point1, point2);
    } else if (metric === 'manhattan') {
      distance = ul.manhattanDistance(point1, point2);
    } else if (metric === 'cosine') {
      distance = 1 - ul.cosineSimilarity(point1, point2); // Convert similarity to distance
    }

    // Draw distance visualization
    if (metric === 'euclidean') {
      // Draw straight line (Euclidean)
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(toScreenX(point1[0]), toScreenY(point1[1]));
      ctx.lineTo(toScreenX(point2[0]), toScreenY(point2[1]));
      ctx.stroke();
    } else if (metric === 'manhattan') {
      // Draw L-shaped path (Manhattan)
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(toScreenX(point1[0]), toScreenY(point1[1]));
      ctx.lineTo(toScreenX(point2[0]), toScreenY(point1[1])); // Horizontal
      ctx.lineTo(toScreenX(point2[0]), toScreenY(point2[1])); // Vertical
      ctx.stroke();
    } else if (metric === 'cosine') {
      // Draw vectors from origin
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toScreenX(0), toScreenY(0));
      ctx.lineTo(toScreenX(point1[0]), toScreenY(point1[1]));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(toScreenX(0), toScreenY(0));
      ctx.lineTo(toScreenX(point2[0]), toScreenY(point2[1]));
      ctx.stroke();
      
      // Draw angle arc
      const angle1 = Math.atan2(point1[1], point1[0]);
      const angle2 = Math.atan2(point2[1], point2[0]);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(toScreenX(0), toScreenY(0), 30, -angle1, -angle2);
      ctx.stroke();
    }

    // Draw points
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(toScreenX(point1[0]), toScreenY(point1[1]), 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(toScreenX(point2[0]), toScreenY(point2[1]), 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('P1', toScreenX(point1[0]), toScreenY(point1[1]) - 15);
    ctx.fillText('P2', toScreenX(point2[0]), toScreenY(point2[1]) - 15);

    // Draw distance value
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const midX = (toScreenX(point1[0]) + toScreenX(point2[0])) / 2;
    const midY = (toScreenY(point1[1]) + toScreenY(point2[1])) / 2;
    ctx.fillText(`Distance: ${distance.toFixed(3)}`, midX, midY - 10);

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const metricNames = {
      euclidean: 'Euclidean Distance (L2)',
      manhattan: 'Manhattan Distance (L1)',
      cosine: 'Cosine Distance'
    };
    ctx.fillText(metricNames[metric] || 'Distance Metric', width / 2, 20);

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
  }, [point1, point2, metric]);

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border-2 border-teal-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-teal-800">
          💡 <strong>Interactive:</strong> Adjust the points and see how different distance metrics measure similarity!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-teal-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Distance Metric
          </label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500"
          >
            <option value="euclidean">Euclidean Distance (L2)</option>
            <option value="manhattan">Manhattan Distance (L1)</option>
            <option value="cosine">Cosine Distance</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Point 1: ({point1[0].toFixed(1)}, {point1[1].toFixed(1)})
            </label>
            <div className="space-y-2">
              <div>
                <label className="text-xs text-gray-600">X:</label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={point1[0]}
                  onChange={(e) => setPoint1([Number(e.target.value), point1[1]])}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-600">Y:</label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={point1[1]}
                  onChange={(e) => setPoint1([point1[0], Number(e.target.value)])}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Point 2: ({point2[0].toFixed(1)}, {point2[1].toFixed(1)})
            </label>
            <div className="space-y-2">
              <div>
                <label className="text-xs text-gray-600">X:</label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={point2[0]}
                  onChange={(e) => setPoint2([Number(e.target.value), point2[1]])}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-600">Y:</label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.1"
                  value={point2[1]}
                  onChange={(e) => setPoint2([point2[0], Number(e.target.value)])}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Metric Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Distance Metrics Explained:</h4>
        <ul className="text-sm text-gray-700 space-y-2">
          <li><strong>Euclidean Distance (L2):</strong> Straight-line distance. d = √(Σ(xᵢ - yᵢ)²). Most common, sensitive to outliers.</li>
          <li><strong>Manhattan Distance (L1):</strong> Sum of absolute differences. d = Σ|xᵢ - yᵢ|. Robust to outliers, used in grid-based paths.</li>
          <li><strong>Cosine Distance:</strong> Measures angle between vectors. d = 1 - cos(θ). Used for text similarity, ignores magnitude.</li>
        </ul>
      </div>
    </div>
  );
}

